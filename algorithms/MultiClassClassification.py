def multiclass_classification():
    print("MULTI-CLASS CLASSIFICATION: \n")

    realOutput = ["cake", "cookie", "cake", "biscuit",
                   "cookie", "cake", "cookie", "biscuit",
                   "biscuit", "cake", "cookie", "cake"]

    computedOutput = ["cookie", "cookie", "cake", "biscuit",
                       "cookie", "biscuit", "cake", "cookie",
                       "biscuit", "cake", "biscuit", "biscuit"]

    computedProbs = [[0.4, 0.3, 0.3], [0.6, 0.2, 0.2], [0.1, 0.3, 0.6], [0.3, 0.5, 0.2],
                              [0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5], [0.8, 0.1, 0.1],
                              [0.2, 0.6, 0.2], [0.1, 0.1, 0.8], [0.3, 0.5, 0.2], [0.4, 0.5, 0.1]]

    # labels = set(realOutput)
    labels = ["cookie", "biscuit", "cake"]


    print("real = " + str(realOutput))
    print("computed = " + str(computedOutput))
    print("probabilities = " + str(computedProbs))
    print("labels = " + str(labels) + "\n")


    computedOutput_probability = computeLabelByProbability(labels, computedProbs)

    confusionMatrix = generate_confusion_matrix(labels, realOutput, computedOutput_probability)

    acc = get_accuracy(confusionMatrix, labels, realOutput)
    prec = get_precision(confusionMatrix, labels)
    rec = get_recall(confusionMatrix, labels)


    print("computed with probabilities = " + str(computedOutput_probability))
    print("confusion matrix : " + str(confusionMatrix) + "\n")
    print('Accuracy: ' + str(acc))
    print('Precision: ' + str(prec))
    print('Recall: ' + str(rec) + "\n")


def computeLabelByProbability(labels, computedProbs):
    result = [0] * len(computedProbs)
    j = 0

    for i in computedProbs:
        index = i.index(max(i))
        result[j] = labels[index]
        j +=1

    return result


def generate_confusion_matrix(labels, realOutput, computedOutput):
    matrix = [[0]*len(labels)]*(len(labels))

    for i in range(len(computedOutput)):
        actual = 0
        computed = 0

        for j in range(len(labels)):
            if realOutput[i] == labels[j]:
                actual = j
            if computedOutput[i] == labels[j]:
                computed = j
        matrix[actual][computed] += 1

    return matrix


def get_accuracy(matrix, labels, realOutput):
    # label-uri nimerite / toate elementele
    total = 0

    for i in range(len(labels)):
        total += matrix[i][i]
    return total / len(realOutput)


def get_precision(matrix, labels):
    # pentru fiecare label, calculam sa vedem cate elemente am avut in total, apoi cate au fost nimerite din acel total
    dictionary = {}

    for i in range(len(labels)):
        total = 0
        for j in range(len(labels)):
            total += matrix[i][j]

        dictionary[labels[i]] = matrix[i][i] / total

    return dictionary

def get_recall(matrix, labels):
    # din toate cele nimerite pe un anumit label, cate dintre ele au fost corecte
    dictionary = {}

    for i in range(len(labels)):
        total = 0
        for j in range(len(labels)):
            total += matrix[j][i]

        dictionary[labels[i]] = matrix[i][i] / total

    return dictionary
