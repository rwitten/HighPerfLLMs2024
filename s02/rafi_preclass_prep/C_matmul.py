import datetime
import jax

import timing_util






######## EXAMPLE 1 START #########

MATMUL_SIZES = [(64000,128), (16000,256), (4000,512), (3000,640), (2000,768), (1000,1024), (250, 2048)]

for num_matmuls, matrix_size in MATMUL_SIZES:
    A = jax.numpy.ones ( (num_matmuls, matrix_size, matrix_size), dtype=jax.numpy.bfloat16)
    B = jax.numpy.ones ( (num_matmuls, matrix_size, matrix_size), dtype=jax.numpy.bfloat16)

    @jax.jit
    def f(X,Y):
        return jax.lax.batch_matmul(X,Y)

    print(f(A,B).shape)
    

    timing_util.simple_timeit(f, A, B, task = 'matmul_' + str(matrix_size))




