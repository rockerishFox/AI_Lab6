from math import sqrt


def single_target_regression():
    print("SINGLE TARGET REGRESSION: \n")

    # neechilibrat:
    # realOutput = [32, 12, 56, 32, 12, 23]
    # computedOutput = [23, 45, 26, 34, 22, 21]

    # echilibrat
    realOutput = [32.5, 12, 56, 78, 12, 23]
    computedOutput = [30, 11, 55.4, 79, 12.2, 21]


    mae = loss_MAE(realOutput, computedOutput)
    mse = loss_MSE(realOutput, computedOutput)

    print("real = " + str(realOutput))
    print("computed = " + str(computedOutput))
    print("MeanAbsoluteError/ L1 Loss: " + str(mae))
    print("MeanSquareError/ Quadratic loss/ L2 Loss: " + str(mse) + "\n")



def loss_MAE(realOutput, computedOutput):
    error = 0
    
    for i in range(len(realOutput)):
        error +=  abs(realOutput[i] - computedOutput[i])
        
    return error / len(realOutput)


def loss_MSE(realOutput, computedOutput):
    error = 0

    for i in range(len(realOutput)):
        error += (realOutput[i] - computedOutput[i]) ** 2

    return error / len(realOutput)
