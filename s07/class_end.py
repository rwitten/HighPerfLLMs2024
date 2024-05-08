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

import sys

BATCH_IN_SEQUENCES = 384
SEQUENCE_LENGTH = 128

VOCAB_DIM = 256
EMBED_DIM = 2048
FF_DIM = 8192

NUM_HEADS = 4
HEAD_DIM = 128

LAYERS = 8

HEAD_DEPTH = 128
NUM_HEADS = 4

LEARNING_RATE = 1e-3

FSDP = 4
TENSOR = 1

LOG_PERIOD = 10

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

    positional_embedding = self.param(
        'positional_embedding',
        nn.with_partitioning(nn.initializers.normal(1), (None, None, "fsdp")),
        (1, SEQUENCE_LENGTH, EMBED_DIM),
        jnp.float32,
    )

    x += positional_embedding


    for i in range(LAYERS):
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

def convert_to_ascii(string_array, max_length):
  result = np.zeros((len(string_array), max_length), dtype=np.uint8)
  for i, string in enumerate(string_array):
    for j, char in enumerate(string):
      if j >= SEQUENCE_LENGTH:
         break
      result[i, j] = char
  return result

def input_to_output(np_array):
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

def main():
    ds = tfds.load('lm1b', split='train', shuffle_files=False)
    ds = ds.batch(BATCH_IN_SEQUENCES)

    rngkey = jax.random.key(0)
    model = OurModel()

    shaped_init = jax.eval_shape( functools.partial(model.init, rngkey), jax.ShapeDtypeStruct((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.uint8))
    state_sharding = nn.get_sharding(shaped_init, mesh)
    _params = jax.jit(model.init, out_shardings = state_sharding)(rngkey, jax.ShapeDtypeStruct((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.uint8))
    
    number_total_params = calculate_num_params(_params)
    number_total_flops = 6 * (BATCH_IN_SEQUENCES * SEQUENCE_LENGTH) * number_total_params
    number_total_flops_per_device = number_total_flops / jax.device_count()

    tx = optax.adam(learning_rate = LEARNING_RATE)
    state = train_state.TrainState.create(
       apply_fn = model.apply,
       params = _params,
       tx = tx
    )

    iter = 0
    static_step = jax.jit(step, static_argnums=1)

    last_step_time = time.time()
    stepnum = 0

    jax.profiler.start_trace("gs://runner-maxtext-logs/rwitten-today-5")

    for example in ds:
       outputs = convert_to_ascii(example['text'].numpy(), SEQUENCE_LENGTH)
       inputs = input_to_output(outputs)

       loss, state = static_step(state, model, inputs, outputs)
       #loss, state = jax.jit(step, static_argnums=1)(state, model, inputs, outputs)
       #loss, state = jax.jit(lambda x,y,z,a : step(x,y,z,a), static_argnums=1)(state, model, inputs, outputs)

       stepnum += 1
       
       if stepnum % LOG_PERIOD == 0:
          new_time = time.time()
          time_elapsed_seconds = (new_time-last_step_time)
          last_step_time = new_time
          print(f"{iter} -> {loss} {time_elapsed_seconds}")
          per_device_tflops_completed_in_interval = number_total_flops_per_device * LOG_PERIOD / 1e12
          print(f"TFLOP/s/device {per_device_tflops_completed_in_interval / time_elapsed_seconds}")

          if stepnum == LOG_PERIOD:
             jax.profiler.stop_trace()
          #sys.exit(0)

          
       

       iter += 1


if __name__ == "__main__":
    main()