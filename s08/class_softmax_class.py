import math

A = [1000, 5, 30]
B = [1, 2, 3]

def simple_softmax(_A, _B):
    maxA = max(A)
    expA = [math.exp(x - maxA) for x in _A] # one read and write from HBM
    denominator = sum(expA) # second read and write from HBM

    output = 0
    for i in range(len(A)):
        output += expA[i] * _B[i] # two reads

    return output / denominator 
    ### Algorithm takes 5 passes back and forth from HBM

def streaming_softmax(_A, _B):
    sum_denominator = 0
    raw_output = 0
    for i in range(len(A)):
        expA = math.exp(_A[i]) # one read
        raw_output += expA * _B[i] # one reads
        sum_denominator += expA

    return raw_output / sum_denominator
    # two reads from HBM

print(simple_softmax(A,B))
print(streaming_softmax(A,B))

