import datetime
import jax

import timing_util

MATRIX_SIZE = 4096
STEPS = 10


######## EXAMPLE 1 START #########

A = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
B = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
C = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))

def matmul(A,B):
    return A@B
jit_matmul = jax.jit(matmul)

def matmul_with_activation(A,B,C):
    return jax.nn.relu(A@B)
jit_matmul_foldin = jax.jit(matmul_with_activation)

timing_util.simple_timeit(matmul, A, B, task = 'matmul')
timing_util.simple_timeit(jit_matmul, A, B, task = 'jit_matmul')

timing_util.simple_timeit(matmul_with_activation, A, B, C, task='matmul_foldin')
timing_util.simple_timeit(jit_matmul_foldin, A, B, C, task = 'jit_matmul_foldin')




