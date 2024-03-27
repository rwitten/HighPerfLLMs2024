import jax
from jax.experimental.pallas.ops import attention as pallas_attention

BATCH = 1
HEADS = 4
SEQUENCE = 2048
HEAD_DIM = 128

Q = jax.random.normal( jax.random.key(0), (BATCH, SEQUENCE, HEADS, HEAD_DIM))
K = jax.random.normal( jax.random.key(1), (BATCH, SEQUENCE, HEADS, HEAD_DIM))
V = jax.random.normal( jax.random.key(2), (BATCH, SEQUENCE, HEADS, HEAD_DIM))


def _attention_by_hand(_Q, _K, _V):
    _W_unnormalized = jax.numpy.einsum('BSHD,BTHD->BHST', _Q, _K)
    mask = 0 * jax.numpy.expand_dims(jax.numpy.expand_dims(jax.numpy.triu(jax.numpy.ones((SEQUENCE,SEQUENCE),jax.numpy.bfloat16),1), axis=0), axis=0)
    _W = jax.nn.softmax(_W_unnormalized - 1e6 * mask)
    out = jax.numpy.einsum('BHST,BTHD->BSHD', _W, _V)
    return out

attention_output = _attention_by_hand(Q,K,V)
pallas_attention_output = pallas_attention.mha_reference(Q,K,V, causal=True, segment_ids=None)


assert jax.numpy.allclose(attention_output, pallas_attention_output, rtol=1e-1, atol=1e-1), "are all close"