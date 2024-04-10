import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state

import flax.linen.attention as attention

import numpy as np

import optax

BATCH_IN_SEQUENCES = 384
SEQUENCE_LENGTH = 128

VOCAB_DIM = 256
EMBED_DIM = 512
FF_DIM = 2048

LAYERS = 4

HEAD_DEPTH = 128
NUM_HEADS = 4

LEARNING_RATE = 1e-3

class OurModel(nn.Module):
  @nn.compact
  def __call__(self, x):
    '''
        x is [BATCH, SEQUENCE]
    '''
    embedding = self.param(
        'embedding',
        nn.initializers.normal(1),
        (VOCAB_DIM, EMBED_DIM),
        jnp.float32,
    )
    x = embedding[x] ##OUTPUT should be [BATCH, SEQUENCE, EMBED]


    for i in range(LAYERS):
      feedforward = self.param(
          'feedforward_' + str(i),
          nn.initializers.lecun_normal(),
          (EMBED_DIM, FF_DIM),
          jnp.float32,
      )
      x = x @ feedforward
      x = jax.nn.relu(x)
      embed = self.param(
          'embed_' + str(i),
          nn.initializers.lecun_normal(),
          (FF_DIM, EMBED_DIM),
          jnp.float32,
      )
      x = x @ embed
      x = jax.nn.relu(x)

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

def main():
    ds = tfds.load('lm1b', split='train', shuffle_files=False)
    ds = ds.batch(BATCH_IN_SEQUENCES)

    rngkey = jax.random.key(0)
    model = OurModel()
    _params = model.init(rngkey, jnp.ones((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.uint8))
    tx = optax.adam(learning_rate = LEARNING_RATE)
    state = train_state.TrainState.create(
       apply_fn = model.apply,
       params = _params,
       tx = tx
    )

    iter = 0
    for example in ds:
       outputs = convert_to_ascii(example['text'].numpy(), SEQUENCE_LENGTH)
       inputs = input_to_output(outputs)
       
       loss, grad = jax.value_and_grad(calculate_loss)(state.params, model, inputs, outputs)
       state = state.apply_gradients(grads = grad)
       print(f"{iter} -> {loss}")
       iter += 1


if __name__ == "__main__":
    main()