import numpy as np
import jax
import timing_util
from functools import partial


A = jax.numpy.ones((16384, 16384), dtype= jax.numpy.bfloat16)
DATA = 1
FSDP = 4
TENSOR = 1

mesh = jax.sharding.Mesh(np.reshape(  jax.devices(), (DATA,FSDP,TENSOR)), ["data", "fsdp", "tensor"])
p = jax.sharding.PartitionSpec("fsdp")
sharding = jax.sharding.NamedSharding(mesh, p)
unsharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
A = jax.device_put(A, sharding)

jax.debug.visualize_array_sharding(A)

@partial(jax.jit, out_shardings = unsharded)
def f(_A):
    return _A

B = f(A)
jax.debug.visualize_array_sharding(B)



time = timing_util.simple_timeit(f,A, task="unshard")
bandwidth_GB_per_s = (2*A.size/1e9)/(time/1e3)
print(f"Bandwith {bandwidth_GB_per_s=}")
