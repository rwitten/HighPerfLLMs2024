
from functools import partial
import jax
import timing_util

MATRIX_SIZE = 16384

A = jax.numpy.ones( (MATRIX_SIZE, MATRIX_SIZE), dtype = jax.numpy.bfloat16)

mesh = jax.sharding.Mesh(jax.devices(), ('ouraxis'))
sharded_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('ouraxis'))
unsharded_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))

A = jax.device_put(A, sharded_sharding)


@partial(jax.jit, out_shardings = unsharded_sharding)
def unshard_array(input):
    return input

average_time_ms = timing_util.simple_timeit(unshard_array, A, task='unshard_array')

achieved_bandwidth_GB_s = (A.size * 2 / 1e9) / (average_time_ms / 1e3)

print(f"{achieved_bandwidth_GB_s=}")




#jax.debug.visualize_array_sharding(unsharded_A)