from functools import partial
import os
import jax
import numpy as np
import timing_util


os.environ['LIBTPU_INIT_ARGS'] = '--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE'

GLOBAL_BATCH = 4096
E = 32768
DIM1_LEN = 2
DIM2_LEN = 2
LAYERS = 4

sharding_schemes = [
    # ((mesh_dim1, mesh_dim2), (activation partition spec), (weight partition spec))
    ((4, 1), ('dim1',), (None,)),  # DP
    ((4, 1), ('dim1',), ('dim1',)),  # FSDP
    ((4, 1), ('dim1',), (None, 'dim1')),  # FSDP
    ((2, 2), ('dim1',), (None, 'dim2')),  # FSDP
    ((4, 1), (None, 'dim1'), ('dim1',)),  # TP
    ((4, 1), (None, 'dim1'), (None, 'dim1')),  # all-gather on activation, not overlappable!
    ((2, 2), ('dim1', 'dim2'), ('dim2', 'dim1')),  # FSDP + TP
]

A = jax.numpy.ones((GLOBAL_BATCH, E), dtype=jax.numpy.bfloat16)

for idx, (mesh_dims, aps, wps) in enumerate(sharding_schemes):
    mesh = jax.sharding.Mesh(np.reshape(jax.devices(), (mesh_dims[0], mesh_dims[1])), ['dim1', 'dim2'])
    activation_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*aps))
    weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*wps))

    @partial(jax.jit, out_shardings=weight_sharding)
    def weight():
        return jax.numpy.ones((E, E), dtype=jax.numpy.bfloat16)

    weights = [weight() for i in range(LAYERS)]

    @partial(jax.jit, out_shardings=activation_sharding)
    def f(_A, _weights):
        for i in range(LAYERS):
            with jax.named_scope(f"layers_{i}"):
                _A = jax.lax.with_sharding_constraint(_A @ _weights[i], activation_sharding)
        return _A

    A_ = jax.device_put(A, activation_sharding)
    time = timing_util.simple_timeit(f, A_, weights, task=f"matmuls{idx}")
    print(f"{aps=}, {wps=}\n")
