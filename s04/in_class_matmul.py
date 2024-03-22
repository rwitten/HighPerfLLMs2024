
from functools import partial
import jax
import timing_util

BATCH_PER_CHIP = 4096
MATRIX_SIZE = 16384
LAYERS = 4

ACTIVATION = jax.numpy.ones( (BATCH_PER_CHIP*jax.device_count(), MATRIX_SIZE), dtype = jax.numpy.bfloat16)

Ws = [jax.numpy.ones( (MATRIX_SIZE, MATRIX_SIZE), dtype = jax.numpy.bfloat16) for i in range(LAYERS)]

mesh = jax.sharding.Mesh(jax.devices(), ('ouraxis'))
activation_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('ouraxis', None))
weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None,'ouraxis'))

ACTIVATION = jax.device_put(ACTIVATION, activation_sharding)
Ws = [jax.device_put(W, weight_sharding) for W in Ws]

@jax.jit
def matmul(_act, _weights):
    for _weight in _weights:
        _act = _act @ _weight
    return _act

average_time_ms = timing_util.simple_timeit(matmul, ACTIVATION, Ws, task='matmul')

#achieved_bandwidth_GB_s = (A.size * 2 / 1e9) / (average_time_ms / 1e3)

#print(f"{achieved_bandwidth_GB_s=}")




#jax.debug.visualize_array_sharding(unsharded_A)