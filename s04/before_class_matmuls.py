import numpy as np
import jax
import timing_util
from functools import partial


BATCH_PER_DEVICE = 1024
GLOBAL_BATCH = BATCH_PER_DEVICE * jax.device_count()
E = 32768
DATA = 1
FSDP = 1
TENSOR = 4
LAYERS = 4

mesh = jax.sharding.Mesh(np.reshape(  jax.devices(), (FSDP,TENSOR)), ["fsdp", "tensor"])
activation_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp", "tensor"))
weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("tensor", "fsdp"))


@partial(jax.jit, out_shardings = activation_sharding)
def activations():
    return jax.numpy.ones( (GLOBAL_BATCH, E), dtype = jax.numpy.bfloat16)

@partial(jax.jit, out_shardings = weight_sharding)
def weight():
    return jax.numpy.ones( (E, E), dtype = jax.numpy.bfloat16)

@partial(jax.jit)
def f(_A, _weights):
    for i in range(LAYERS):
        _A *= _A @ _weights[i]
    return _A

A = activations()
weights = [weight() for i in range(LAYERS)]

time = timing_util.simple_timeit(f,A,weights,task="matuls")

flops_per_device = (2*BATCH_PER_DEVICE*E*E * LAYERS)
TFLOP_per_sec = (flops_per_device/10**12) / (time/10**3)
print(f"{TFLOP_per_sec=}")
