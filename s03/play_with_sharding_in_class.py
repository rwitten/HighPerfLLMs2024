import numpy as np
import jax

A = jax.numpy.ones((1024,1024, 128))


mesh = jax.sharding.Mesh(np.reshape(  jax.devices(), (1,4)), ["myaxis1", "myaxis2"])
p1 = jax.sharding.PartitionSpec( "myaxis2", "myaxis1")

sharding1 = jax.sharding.NamedSharding(mesh, p1)

sharded_A1 = jax.device_put(A, sharding1)



print(f"{sharded_A1.shape=} {sharded_A1.addressable_shards[0].data.shape=}")

#jax.debug.visualize_array_sharding(output)