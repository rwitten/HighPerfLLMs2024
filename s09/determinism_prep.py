import math
import random
import jax

A1 = [math.sin(x) for x in list(range(100000))]

A2 = [math.sin(x) for x in list(range(100000))]
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



def int8_sum(input_arr):
    input_arr_jax = jax.numpy.asarray(input_arr, dtype=jax.numpy.float32)
    absmax = jax.numpy.max(jax.numpy.abs(input_arr_jax))
    scaled = jax.numpy.asarray(input_arr_jax / absmax * 127, dtype = jax.numpy.int8)
    output_sum = jax.numpy.sum(scaled)
    return output_sum * absmax / 127

print(f"{sum(A1)=}")
print(f"{sum(A2)=}")

print(f"{int8_sum(A1)=}")
print(f"{int8_sum(A2)=}")

print(f"{jax_sum(A1)=}")
print(f"{jax_sum(A2)=}")

print(f"{jax_middle_sum(A1)=}")
print(f"{jax_middle_sum(A2)=}")

print(f"{jax_bad_sum(A1)=}")
print(f"{jax_bad_sum(A2)=}")
