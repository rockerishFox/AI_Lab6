from math import sqrt


def multi_target_regression():
    print("MULTI TARGET REGRESSION: \n")

    # neechilibrat:
    # realOutput = [[12, 12, 12], [13, 65, 21.1], [23, 45, 89]]
    # computedOutput = [[14, 10, 15], [11, 66, 23], [25, 43, 90]]

    # echilibrat
    realOutput = [[12, 12, 12], [13, 65, 21.1], [23, 45, 89]]
    computedOutput = [[12.1, 11.5, 12], [12.4, 65.4, 21.5], [22, 45.4, 88]]

    mae = calculate_mae(realOutput, computedOutput)
    rmse = calculate_rmse(realOutput, computedOutput)

    print("real = " + str(realOutput))
    print("computed = " + str(computedOutput))
    print("MeanAbsoluteError: " + str(mae))
    print("RootMeanSquareError: " + str(rmse) + "\n")


    #\033[1;34;48m

# MAE = 1/n * ( SUM ( |real-computed| ) )
def calculate_mae(realOutput, computedOutput):
    error = 0
    for i in range(len(realOutput)):
        sum = 0

        for j in range(len(realOutput[i])):
            sum += abs(realOutput[i][j] - computedOutput[i][j])

        error += sum / len(realOutput[i])

    return error / len(realOutput)


def calculate_rmse(realOutput, computedOutput):
    error = 0
    for i in range(len(realOutput)):
        sum = 0

        for j in range(len(realOutput[i])):
            sum += (realOutput[i][j] - computedOutput[i][j]) ** 2

        error += sqrt(sum / len(realOutput[i]))

    return error / len(realOutput)
