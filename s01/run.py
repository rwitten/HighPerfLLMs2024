import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state

import flax.linen.attention as attention

import numpy as np

import optax

BATCH_IN_SEQUENCES = 256
SEQUENCE_LENGTH = 128

VOCAB_DIM = 256
EMBED_DIM = 512
FF_DIM = 2048

LAYERS = 4

HEAD_DEPTH = 128
NUM_HEADS = 4

LEARNING_RATE = 1e-3


###### START MODEL ######

class OurModel(nn.Module):
  def layer_impl(self, x):
     return x

  @nn.compact
  def __call__(self, input_tokens):
    '''
        input_tokens is [BATCH, SEQUENCE]
    '''
    embedding = self.param(
        'embedding',
        nn.initializers.normal(1),
        (VOCAB_DIM, EMBED_DIM),
        jnp.float32,
    )

    x = jnp.asarray(embedding)[input_tokens] # BATCH, SEQUENCE, EMBED

    pos_embedding = self.param(
        'pos_embedding',
        nn.initializers.normal(1),
        (1, SEQUENCE_LENGTH, EMBED_DIM),
        jnp.float32,
    )

    x += jnp.asarray(pos_embedding)

    for i in range(LAYERS):
        layer_input = x
        emb2ff = self.param(
            'emb2ff_' + str(i),
            nn.initializers.lecun_normal(),
            (EMBED_DIM, FF_DIM),
            jnp.float32,
        )

        x = x @ jnp.asarray(emb2ff)
        x = jax.nn.relu(x)
        ff2emb = self.param(
            'ff2emb_' + str(i),
            nn.initializers.lecun_normal(),
            (FF_DIM, EMBED_DIM),
            jnp.float32,
        )
        x = x @ jnp.asarray(ff2emb)
        x = jax.nn.relu(x)

        q_proj = self.param(
            'q_proj_' + str(i),
            nn.initializers.lecun_normal(),
            (EMBED_DIM, NUM_HEADS, HEAD_DEPTH),
            jnp.float32,
        )
        q = jnp.einsum("BSE,END->BSND", x, jnp.asarray(q_proj))
        
        k_proj = self.param(
            'k_proj_' + str(i),
            nn.initializers.lecun_normal(),
            (EMBED_DIM, NUM_HEADS, HEAD_DEPTH),
            jnp.float32,
        )
        k = jnp.einsum("BSE,END->BSND", x, jnp.asarray(k_proj))

        v_proj = self.param(
            'v_proj_' + str(i),
            nn.initializers.lecun_normal(),
            (EMBED_DIM, NUM_HEADS, HEAD_DEPTH),
            jnp.float32,
        )
        v = jnp.einsum("BSE,END->BSND", x, jnp.asarray(v_proj))

        post_attention = attention.dot_product_attention(q, k, v) #### WARNING: THIS IS CHEATING A LITTLE BECAUSE OF CAUSALITY
        post_attention = jax.numpy.reshape(post_attention, (BATCH_IN_SEQUENCES, SEQUENCE_LENGTH, NUM_HEADS * HEAD_DEPTH))

        out_proj = self.param(
            'out_proj_' + str(i),
            nn.initializers.lecun_normal(),
            (NUM_HEADS * HEAD_DEPTH, EMBED_DIM),
            jnp.float32,
        )

        x = post_attention @ jnp.asarray(out_proj)
        x += layer_input
        
    
    emb2vocab = self.param(
        'emb2vocab' + str(i),
        nn.initializers.lecun_normal(),
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

    # Build your input pipeline
    ds = ds.batch(BATCH_IN_SEQUENCES).prefetch(tf.data.AUTOTUNE)
    return ds

def process_example(raw_example):
    numpy_strings = raw_example['text'].numpy()
    ascii_array_input = convert_to_ascii(numpy_strings, SEQUENCE_LENGTH)
    ascii_array_output = 0 * np.empty_like(ascii_array_input)
    ascii_array_output[:,0:SEQUENCE_LENGTH-1] = ascii_array_input[:, 1:SEQUENCE_LENGTH]
    return {"input" : jnp.asarray(ascii_array_input), "output" : jnp.asarray(ascii_array_output)}

############ DATA LOADING END ############


def calculate_loss(params, example, model):
   logits = model.apply(params, example["input"])
   n_classes = logits.shape[-1]
   loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(example["output"], n_classes)).mean()
   return loss

def main():
    rngkey = jax.random.key(0)
    ds = build_dataset()
    model = OurModel()
    tx = optax.adam(learning_rate=LEARNING_RATE)
    params = model.init(rngkey, jax.numpy.ones((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.int8))

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

    step = 0
    for raw_example in ds:
        example = process_example(raw_example)
        loss, grad = jax.value_and_grad(calculate_loss, argnums = 0)(state.params, example, model)
        state = state.apply_gradients(grads=grad)
        print(f"{step=}, {float(loss)=}")
        step += 1
        
if __name__ == "__main__":
    main()