import math
import random
import jax

A1 = [math.sin(x) for x in list(range(10000))]

A2 = [math.sin(x) for x in list(range(10000))]
random.shuffle(A2)

