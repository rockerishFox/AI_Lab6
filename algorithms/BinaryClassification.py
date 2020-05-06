import math

def binary_classification():
    print("BINARY CLASSIFICATION: \n")

    realOutput = [0, 0, 1, 0, 1, 0]
    computedOutput = [[0.6, 0.4], [0.4, 0.6], [0.6, 0.4], [0.5, 0.5], [0.6, 0.4], [0.9, 0.1]]

    loss = getloss(realOutput, computedOutput)

    print("Categorical Cross-Entropy Loss: " + str(loss))


def getloss(real, computed):
    # −( y log(p) + (1−y) log(1−p) )
    suma = 0
    for x, y in zip(real,computed):
        suma += math.log(y[x])

    return -suma/len(real)

