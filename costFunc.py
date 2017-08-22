import numpy as np

def costCal(ypredict, ylabel):
    if ypredict.shape != ylabel.shape:
        print('error in costCal: two results are with different size\n')
        exit(1)
    sampleNum = ylabel.shape[0]
    cost = np.sum(np.sum(((ypredict - ylabel)**2 / 2), axis=1), axis=0) / sampleNum
    return cost
