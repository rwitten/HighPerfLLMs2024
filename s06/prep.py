import functools
import tensorflow as tf
import tensorflow_datasets as tfds

import time

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state

import flax.linen.attention as attention

import numpy as np

import optax

from jax.experimental.pallas.ops import attention as pallas_attention

import sys

PRINT_TIMER = 10

FSDP = 4
TENSOR = 1


BATCH_IN_SEQUENCES = 1024
SEQUENCE_LENGTH = 128

VOCAB_DIM = 256
EMBED_DIM = 2048
FF_DIM = 8192

LAYERS = 8

HEAD_DEPTH = 128
NUM_HEADS = 8

LEARNING_RATE = 1e-5

###### START MODEL ######

class OurModel(nn.Module):
  @nn.compact
  def __call__(self, input_tokens):
    '''
        input_tokens is [BATCH, SEQUENCE]
    '''
    embedding = self.param(
        'embedding',
        nn.with_partitioning(nn.initializers.normal(1), ('tp', 'fsdp')),
        (VOCAB_DIM, EMBED_DIM),
        jnp.float32,
    )

    x = jnp.asarray(embedding)[input_tokens] # BATCH, SEQUENCE, EMBED

    pos_embedding = self.param(
        'pos_embedding',
        nn.with_partitioning(nn.initializers.normal(1), ('tp', 'fsdp')),
        (1, SEQUENCE_LENGTH, EMBED_DIM),
        jnp.float32,
    )

    x += jnp.asarray(pos_embedding)

    for i in range(LAYERS):
        layer_input = x
        emb2ff = self.param(
            'emb2ff_' + str(i),
            nn.with_partitioning(nn.initializers.lecun_normal(), ('tp', 'fsdp')),
            (EMBED_DIM, FF_DIM),
            jnp.float32,
        )

        x = x @ jnp.asarray(emb2ff)
        x = jax.nn.relu(x)
        ff2emb = self.param(
            'ff2emb_' + str(i),
            nn.with_partitioning(nn.initializers.lecun_normal(), ('tp', 'fsdp')),
            (FF_DIM, EMBED_DIM),
            jnp.float32,
        )
        x = x @ jnp.asarray(ff2emb)
        x = jax.nn.relu(x)

        q_proj = self.param(
            'q_proj_' + str(i),
            nn.with_partitioning(nn.initializers.lecun_normal(), ('tp', 'fsdp')),
            (EMBED_DIM, NUM_HEADS, HEAD_DEPTH),
            jnp.float32,
        )
        q = jnp.einsum("BSE,END->BSND", x, jnp.asarray(q_proj))
        
        k_proj = self.param(
            'k_proj_' + str(i),
            nn.with_partitioning(nn.initializers.lecun_normal(), ('tp', 'fsdp')),
            (EMBED_DIM, NUM_HEADS, HEAD_DEPTH),
            jnp.float32,
        )
        k = jnp.einsum("BSE,END->BSND", x, jnp.asarray(k_proj))

        v_proj = self.param(
            'v_proj_' + str(i),
            nn.with_partitioning(nn.initializers.lecun_normal(), ('tp', 'fsdp')),
            (EMBED_DIM, NUM_HEADS, HEAD_DEPTH),
            jnp.float32,
        )
        v = jnp.einsum("BSE,END->BSND", x, jnp.asarray(v_proj))

        post_attention = pallas_attention.mha_reference(q,k,v, causal=True, segment_ids=None)

        post_attention = jax.numpy.reshape(post_attention, (BATCH_IN_SEQUENCES, SEQUENCE_LENGTH, NUM_HEADS * HEAD_DEPTH))

        out_proj = self.param(
            'out_proj_' + str(i),
            nn.with_partitioning(nn.initializers.lecun_normal(), ('tp', 'fsdp')),
            (NUM_HEADS * HEAD_DEPTH, EMBED_DIM),
            jnp.float32,
        )

        x = post_attention @ jnp.asarray(out_proj)
        x += layer_input
        
    
    emb2vocab = self.param(
        'emb2vocab' + str(i),
        nn.with_partitioning(nn.initializers.lecun_normal(), ('fsdp', 'tp')),
        (EMBED_DIM, VOCAB_DIM),
        jnp.float32,
    )

    x = x @ jnp.asarray(emb2vocab) # should output (BATCH, SEQUENCE, VOCAB)
    return x
    
###### END MODEL ######


############ DATA LOADING START ############

def convert_to_ascii(string_array, max_length):
  result = np.zeros((len(string_array), max_length), dtype=np.uint8)
  for i, string in enumerate(string_array):
    for j, char in enumerate(string):
      if j >= SEQUENCE_LENGTH:
         break
      result[i, j] = char
  return result

def build_dataset():
    # Construct a tf.data.Dataset
    ds = tfds.load('lm1b', split='train', shuffle_files=False)
    ds = ds.repeat(1)

    # Build your input pipeline
    ds = ds.batch(BATCH_IN_SEQUENCES).prefetch(tf.data.AUTOTUNE)
    return ds

def process_example(raw_example):
    numpy_strings = raw_example['text'].numpy()
    ascii_array_output = convert_to_ascii(numpy_strings, SEQUENCE_LENGTH)
    ascii_array_input = 0 * np.empty_like(ascii_array_output)
    ascii_array_input[:,1:SEQUENCE_LENGTH] = ascii_array_input[:, 0:SEQUENCE_LENGTH-1]
    return {"input" : jnp.asarray(ascii_array_input), "output" : jnp.asarray(ascii_array_output)}

############ DATA LOADING END ############


def calculate_loss(params, example, model):
    logits = model.apply(params, example["input"])
    n_classes = logits.shape[-1]
    loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(example["output"], n_classes)).mean()

    return loss

def take_step(state, example, model):
    loss, grad = jax.value_and_grad(calculate_loss, argnums = 0)(state.params, example, model)
    state = state.apply_gradients(grads=grad)
    return state, loss


def count_params(p):
   def f(x):
        if isinstance(x, jax.Array):
            return x.size
        else:
            return 0
   all_p = jax.tree_util.tree_map(f, p)
   return jax.tree_util.tree_reduce(lambda a,b : a+b, all_p)

def count_params_per_chip(p):
   def f(x):
        if isinstance(x, jax.Array):
            return x.addressable_shards[0].data.size
        else:
            return 0

   all_p = jax.tree_util.tree_map(f, p)
   return jax.tree_util.tree_reduce(lambda a,b : a+b, all_p)

def main():
    mesh = jax.sharding.Mesh(np.reshape(  jax.devices(), (FSDP,TENSOR)), [ "fsdp", "tp"])

    rngkey = jax.random.key(0)
    ds = build_dataset()
    model = OurModel()
    tx = optax.adam(learning_rate=LEARNING_RATE)
    abstract_params = jax.eval_shape(functools.partial(model.init, rngkey), jax.ShapeDtypeStruct((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.int8))
    state_sharding = nn.get_sharding(abstract_params, mesh)
    params = jax.jit(model.init, out_shardings = state_sharding)(rngkey, jax.ShapeDtypeStruct((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.int8))

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

    num_state_billions = count_params(state) / 1e9
    num_state_per_chip_billions = count_params_per_chip(state) / 1e9
    print(f"{num_state_billions=} {num_state_per_chip_billions=}")

    total_flops = (6*(num_state_billions*1e9/3)*BATCH_IN_SEQUENCES*SEQUENCE_LENGTH)
    print(f"{total_flops=}")


    step_function = jax.jit(take_step, static_argnums=2)

    step = 0

    last_time = time.time()
    for raw_example in ds:
        example = process_example(raw_example)
        example = jax.device_put(example, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp", "tp")))
        state, loss = step_function(state, example, model)

        if step > 0 and step % PRINT_TIMER == 0:
            new_time = time.time()
            time_elapsed_seconds = (new_time-last_time)
            tflop_per_second = PRINT_TIMER * total_flops / time_elapsed_seconds / 1e12
            print(f"{step=}, {float(loss)=} {new_time=} {tflop_per_second=} {time_elapsed_seconds=}")
            last_time = new_time

        step += 1
        
if __name__ == "__main__":
    main()