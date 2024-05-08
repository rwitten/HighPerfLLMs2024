import math

A = [0, 1000, 100]
B = [1, 2, 2]

def softmax_naive(A, B):
    expA = [math.exp(x - max(A)) for x in A]
    normalizer = sum(expA)

    print(expA)
    output = 0
    for i in range(3):
        output += B[i] * expA[i] / normalizer 
    return output


def rolling_softmax_naive(A, B):
    denominator = 0
    output = 0

    for i in range(3):
        addition_to_denominator = math.exp(A[i])
        output += B[i] * addition_to_denominator
        denominator += addition_to_denominator

    return output / denominator

print(softmax_naive(A,B))
print(rolling_softmax_naive(A,B))
