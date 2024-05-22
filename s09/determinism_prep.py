import math
import random
import jax

A1 = [math.sin(x) for x in list(range(10000))]

A2 = [math.sin(x) for x in list(range(10000))]
random.shuffle(A2)

def jax_sum(input_arr):
    arr = jax.numpy.asarray(input_arr, dtype=jax.numpy.float32)
    output = jax.numpy.sum(arr)
    return output

def jax_middle_sum(input_arr):
    arr = jax.numpy.asarray(input_arr, dtype=jax.numpy.bfloat16)
    output = jax.numpy.sum(arr)
    return output

def jax_bad_sum(input_arr):
    accum = jax.numpy.zeros( (), dtype=jax.numpy.bfloat16)
    for i in input_arr:
        accum += jax.numpy.asarray(i, dtype=jax.numpy.bfloat16)
        accum = jax.numpy.asarray(accum, dtype=jax.numpy.bfloat16)

    return accum

print(f"{sum(A1)=}")
print(f"{sum(A2)=}")

print(f"{jax_sum(A1)=}")
print(f"{jax_sum(A2)=}")

print(f"{jax_middle_sum(A1)=}")
print(f"{jax_middle_sum(A2)=}")

print(f"{jax_bad_sum(A1)=}")
print(f"{jax_bad_sum(A2)=}")
