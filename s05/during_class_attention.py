import jax
from jax.experimental.pallas.ops import attention as pallas_attention

BATCH = 1
HEADS = 4
SEQUENCE = 2048
HEAD_DIM = 128

Q = jax.random.normal( jax.random.key(0), (BATCH, SEQUENCE, HEADS, HEAD_DIM))
K = jax.random.normal( jax.random.key(1), (BATCH, SEQUENCE, HEADS, HEAD_DIM))
V = jax.random.normal( jax.random.key(2), (BATCH, SEQUENCE, HEADS, HEAD_DIM))
