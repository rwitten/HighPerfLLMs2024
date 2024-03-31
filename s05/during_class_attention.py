import jax
from jax.experimental.pallas.ops import attention as pallas_attention

BATCH = 1
HEADS = 1
SEQUENCE = 2048
HEAD_DIM = 128

Q = jax.random.normal( jax.random.key(0), (BATCH, SEQUENCE, HEADS, HEAD_DIM))
K = jax.random.normal( jax.random.key(1), (BATCH, SEQUENCE, HEADS, HEAD_DIM))
V = jax.random.normal( jax.random.key(2), (BATCH, SEQUENCE, HEADS, HEAD_DIM))



def attention_ourselves(_Q, _K, _V):
    _weights_unnormalized = jax.numpy.einsum("BSHD,BTHD->BHST", _Q, _K)
    _weights_unnormalized_to_zero_out = jax.numpy.triu( jax.numpy.ones((SEQUENCE,SEQUENCE), jax.numpy.bfloat16), 1)
    _weights = jax.nn.softmax(_weights_unnormalized - 1e6 * _weights_unnormalized_to_zero_out)  ### Creating something of size (B,HEADS, SEQUENCE, SEQUENCE)
    #print(f"{_weights.size=}")
    output = jax.numpy.einsum("BHST,BTHD->BSHD", _weights, _V)

    return output

attn_ourselves_value = attention_ourselves(Q,K,V)
attn_value = pallas_attention.mha_reference(Q, K, V,  segment_ids=None, causal=True)

assert jax.numpy.allclose(attn_ourselves_value, attn_value, atol=1e-1, rtol=1e-1)