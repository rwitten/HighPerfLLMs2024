import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state

import functools

import flax.linen.attention as attention

import numpy as np

import optax

import time

import orbax.checkpoint as ocp


BATCH_IN_SEQUENCES = 1
SEQUENCE_LENGTH = 128

VOCAB_DIM = 256
EMBED_DIM = 1024
FF_DIM = 4096

HEAD_DIM = 128

LAYERS = 8 ### go to 1 sometimes?

HEAD_DEPTH = 128
NUM_HEADS = 4

LEARNING_RATE = 1e-6

FSDP = 1
TENSOR = 4

LOG_PERIOD = 10
CHECKPOINT_PERIOD = 1000

mesh = jax.sharding.Mesh(np.reshape(  jax.devices(), (FSDP,TENSOR)), ["fsdp", "tp"])
desired_embedding_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp", None, "tp"))  # apply this to things that are BATCH, SEQUENCE, EMBED


def attention_ourselves(_Q, _K, _V):
    _weights_unnormalized = jax.numpy.einsum("BSHD,BTHD->BHST", _Q, _K)
    _weights_unnormalized_to_zero_out = jax.numpy.triu( jax.numpy.ones((SEQUENCE_LENGTH,SEQUENCE_LENGTH), jax.numpy.bfloat16), 1)
    _weights = jax.nn.softmax(_weights_unnormalized - 1e6 * _weights_unnormalized_to_zero_out)  ### Creating something of size (B,HEADS, SEQUENCE, SEQUENCE)
    #print(f"{_weights.size=}")
    output = jax.numpy.einsum("BHST,BTHD->BSHD", _weights, _V)

    return output

class OurModel(nn.Module):
  @nn.compact
  def __call__(self, x):
    '''
        x is [BATCH, SEQUENCE]
    '''
    embedding = self.param(
        'embedding',
        nn.with_partitioning(nn.initializers.normal(1), ("tp", "fsdp")),
        (VOCAB_DIM, EMBED_DIM),
        jnp.float32,
    )
    x = embedding[x] ##OUTPUT should be [BATCH, SEQUENCE, EMBED]




    for i in range(LAYERS):

      x = nn.LayerNorm( name="layer_norm_" + str(i),)(x)

      positional_embedding = self.param(
        'positional_embedding_' + str(i),
        nn.with_partitioning(nn.initializers.normal(1), (None, None, "fsdp")),
        (1, SEQUENCE_LENGTH, EMBED_DIM),
        jnp.float32,
      )

      x += positional_embedding

      x = jax.lax.with_sharding_constraint(x, desired_embedding_sharding)
      feedforward = self.param(
          'feedforward_' + str(i),
          nn.with_partitioning(nn.initializers.lecun_normal(), ('fsdp', 'tp')),
          (EMBED_DIM, FF_DIM),
          jnp.float32,
      )
      x = x @ feedforward
      x = jax.nn.relu(x)
      x = jax.lax.with_sharding_constraint(x, desired_embedding_sharding)
      embed = self.param(
          'embed_' + str(i),
          nn.with_partitioning(nn.initializers.lecun_normal(), ('tp', 'fsdp')),
          (FF_DIM, EMBED_DIM),
          jnp.float32,
      )
      x = x @ embed
      x = jax.nn.relu(x)

      q_proj = self.param(
          'qproj_' + str(i),
          nn.with_partitioning(nn.initializers.lecun_normal(), ('fsdp', 'tp')),
          (EMBED_DIM, NUM_HEADS, HEAD_DIM),
          jnp.float32,
      )
      q = jnp.einsum("BSE,EHD->BSHD",x, q_proj )

      k_proj = self.param(
          'kproj_' + str(i),
          nn.with_partitioning(nn.initializers.lecun_normal(), ('fsdp', 'tp')),
          (EMBED_DIM, NUM_HEADS, HEAD_DIM),
          jnp.float32,
      )
      k = jnp.einsum("BSE,EHD->BSHD",x, k_proj )

      v_proj = self.param(
          'vproj_' + str(i),
          nn.with_partitioning(nn.initializers.lecun_normal(), ('fsdp', 'tp')),
          (EMBED_DIM, NUM_HEADS, HEAD_DIM),
          jnp.float32,
      )
      v = jnp.einsum("BSE,EHD->BSHD",x, v_proj )

      o = attention_ourselves(q,k,v)

      o_proj = self.param(
          'oproj_' + str(i),
          nn.with_partitioning(nn.initializers.lecun_normal(), ('fsdp', 'tp')),
          (NUM_HEADS, HEAD_DIM, EMBED_DIM),
          jnp.float32,
      )
      x = jnp.einsum("BSHD,HDE->BSE",o, o_proj )
      x = jax.lax.with_sharding_constraint(x, desired_embedding_sharding)

    return x @ embedding.T


def numpy_to_string(numpy_arr):
    return "".join([chr(item) for item in numpy_arr])

def convert_to_ascii(string_array, max_length):
  result = np.zeros((len(string_array), max_length), dtype=np.uint8)
  for i, string in enumerate(string_array):
    for j, char in enumerate(string):
      if j >= SEQUENCE_LENGTH:
         break
      result[i, j] = ord(char)
  return result

def output_to_input(np_array):
   zero_array = np.zeros( (BATCH_IN_SEQUENCES,SEQUENCE_LENGTH), dtype = jnp.uint8)
   zero_array[:, 1:SEQUENCE_LENGTH] = np_array[:, 0:SEQUENCE_LENGTH-1]
   return zero_array

def calculate_loss(params, model, inputs, outputs):
   proposed_outputs = model.apply(params, inputs)
   one_hot = jax.nn.one_hot(outputs, VOCAB_DIM)
   loss = optax.softmax_cross_entropy(proposed_outputs, one_hot)
   return jnp.mean(loss)


def step(state, model, inputs, outputs):
   loss, grad = jax.value_and_grad(calculate_loss)(state.params, model, inputs, outputs)
   state = state.apply_gradients(grads = grad)
   return loss, state

def calculate_num_params(pyt):
   sizes = jax.tree_util.tree_map(lambda x: x.size, pyt)
   return jax.tree_util.tree_reduce(lambda x, y: x+y, sizes)

def visualize_input_to_output(input_string, output_string):
    for i in range(30):
        print(f"{i}: {input_string[:i]} -> `{output_string[i]}`")

def main():
    ds = tfds.load('lm1b', split='train', shuffle_files=False)
    ds = ds.batch(BATCH_IN_SEQUENCES)

    rngkey = jax.random.key(0)
    model = OurModel()

    shaped_init = jax.eval_shape( functools.partial(model.init, rngkey), jax.ShapeDtypeStruct((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.uint8))
    state_sharding = nn.get_sharding(shaped_init, mesh)
    _params = jax.jit(model.init, out_shardings = state_sharding)(rngkey, jax.ShapeDtypeStruct((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.uint8))
    
    number_total_params = calculate_num_params(_params)
    print(f"Number total params {number_total_params/1e6} million")
    number_total_flops = 6 * (BATCH_IN_SEQUENCES * SEQUENCE_LENGTH) * number_total_params
    number_total_flops_per_device = number_total_flops / jax.device_count()

    tx = optax.adam(learning_rate = LEARNING_RATE)
    state = train_state.TrainState.create(
       apply_fn = model.apply,
       params = _params,
       tx = tx
    )

    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore('/home/rwitten/class_checkpoints/checkpoint_0078000', args=ocp.args.StandardRestore(abstract_state))

    text = np.zeros( (1, SEQUENCE_LENGTH), dtype = np.int32)
    NUM_TOKENS = 30
    for i in range(NUM_TOKENS):
        logits = model.apply(state.params, text) # here is my probability distribution! [BATCH, SEQUENCE, VOCAB]
        new_tokens = jax.numpy.argmax(logits, axis=2)
        text[0, i+1] = new_tokens[0, i]
        breakpoint()

    output_string = ""
    for i in range(NUM_TOKENS):
        output_string += chr(text[0,i])

    print(output_string)

if __name__ == "__main__":
    main()