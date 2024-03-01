import datetime
import jax

import timing_util

MATRIX_SIZE = 32768
STEPS = 10


######## EXAMPLE 1 START #########

A = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
B = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))

jax.profiler.start_trace("/tmp/wrong_way_3")

s = datetime.datetime.now()
for i in range(STEPS):
    O = A + B
e = datetime.datetime.now()

jax.profiler.stop_trace()


print( f"Straight addition takes {(e-s).total_seconds()/STEPS:.4f}")

######## EXAMPLE 1 END #########


######## EXAMPLE 2 START #########

A = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
B = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
C = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))

s = datetime.datetime.now()
for i in range(STEPS):
    O = A + B + C
e = datetime.datetime.now()

print( f"Two additions takes {(e-s).total_seconds()/STEPS:.4f}")


######## EXAMPLE 2 END #########

######## EXAMPLE 3 START #########

def f3(X,Y,Z):
    return X+Y+Z

f3_jit = jax.jit(f3)



A = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
B = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
C = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))


s = datetime.datetime.now()
for i in range(STEPS):
    O = f3(A, B, C)
e = datetime.datetime.now()


print( f"Two additions takes {(e-s).total_seconds()/STEPS:.4f} with magic JIT sauce")

######## EXAMPLE 3 END #########

###### EXAMPLE 4 START #####

def f2(A,B):
    return A + B 
f2_jit = jax.jit(f2)


timing_util.simple_timeit(f2, A, B, task = "f2")
timing_util.simple_timeit(f2_jit, A, B, task = "f2_jit")


timing_util.simple_timeit(f3, A, B, C, task = "f3")
timing_util.simple_timeit(f3_jit, A, B, C, task = "f3_jit")


###### EXAMPLE 4 END #####
