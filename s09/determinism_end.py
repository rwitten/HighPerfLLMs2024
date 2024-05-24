import math
import random
import jax

A1 = [math.sin(x) for x in list(range(10000))]

A2 = [math.sin(x) for x in list(range(10000))]
random.shuffle(A2)

print(f"A1-sum {sum(A1)} {sum(A1)}")
print(f"A2-sum {sum(A2)} {sum(A2)}")

def jax_sum_f32(input_vec):
    accumulator = jax.numpy.zeros( (1), dtype = jax.numpy.float32)
    for x in input_vec:
        accumulator += jax.numpy.asarray(x, dtype = jax.numpy.float32)
        accumulator = jax.numpy.asarray(accumulator, dtype = jax.numpy.float32)
    return accumulator

#print(f"jax_sum {jax_sum_f32(A1)} {jax_sum_f32(A2)}")

def jax_sum_accumf32_inputbf16(input_vec):
    accumulator = jax.numpy.zeros( (1), dtype = jax.numpy.float32)
    for x in input_vec:
        accumulator += jax.numpy.asarray(x, dtype = jax.numpy.bfloat16)
        accumulator = jax.numpy.asarray(accumulator, dtype = jax.numpy.float32)
    return accumulator

#print(f"jax_sum_accumf32_inputbf16 {jax_sum_accumf32_inputbf16(A1)} {jax_sum_accumf32_inputbf16(A2)}")

def jax_sum_accumbf16_inputbf16(input_vec):
    accumulator = jax.numpy.zeros( (1), dtype = jax.numpy.bfloat16)
    for x in input_vec:
        accumulator += jax.numpy.asarray(x, dtype = jax.numpy.bfloat16)
        accumulator = jax.numpy.asarray(accumulator, dtype = jax.numpy.bfloat16)
    return accumulator

#print(f"jax_sum_accumbf16_inputbf16 {jax_sum_accumbf16_inputbf16(A1)} {jax_sum_accumbf16_inputbf16(A2)}")

def jax_sum_accumbf16_inputf32(input_vec):
    accumulator = jax.numpy.zeros( (1), dtype = jax.numpy.bfloat16)
    for x in input_vec:
        accumulator += jax.numpy.asarray(x, dtype = jax.numpy.float32)
        accumulator = jax.numpy.asarray(accumulator, dtype = jax.numpy.bfloat16)
    return accumulator

print(f"jax_sum_accumbf16_inputf32 {jax_sum_accumbf16_inputf32(A1)} {jax_sum_accumbf16_inputf32(A2)}")



