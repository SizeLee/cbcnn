import numpy as np

def classifyAccuracyRate(ypredict, ylabel):
    sampleNum = ylabel.shape[0]
    right = 0

    ypredictMax = np.argmax(ypredict, axis=1)
    ylabel = np.argmax(ylabel, axis=1)
    for i in range(sampleNum):
        if ypredictMax[i] == ylabel[i]:
            right += 1;

    return right/sampleNum

def AccuracyRate(ypredictArgmaxList, ylabelArgmaxList):
    sampleNum = len(ylabelArgmaxList)
    right = 0
    for i in range(sampleNum):
        if ypredictArgmaxList[i] == ylabelArgmaxList[i]:
            right += 1

    accuracy = right/sampleNum

    return accuracy
