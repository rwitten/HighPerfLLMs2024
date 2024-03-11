import numpy as np
import jax

#jax.lax.with_sharding_constraint
mesh = jax.sharding.Mesh(np.reshape(jax.devices(), (2,2)), ('axis1', 'axis2'))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None,('axis1', 'axis2')))


k = jax.random.key(0)
A = jax.random.normal(k, (8,4))
A = jax.device_put(A, sharding)

print(A.addressable_shards[0].data.shape)

jax.debug.visualize_array_sharding(A)