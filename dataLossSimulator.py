import numpy as np
import random
import myLoadData
class dataLossSimulator:
    # originDim = 0 ###number of dim of loss range, first originDim of data exist loss
    # lossRate = 0.
    # lossDimEachSample = 0
    # lossSetValue = 0.
    def __init__(self, originDim, lossRate, lossSetValue = 0):
        self.originDim = 0  ###number of dim of loss range, first originDim of data exist loss
        self.lossRate = 0.
        self.lossDimEachSample = 0
        self.lossSetValue = 0.

        if lossRate < 0 or lossRate > 1:
            print('Error in dataLossSimulator: loss Dim is larger than originDim\n')
            exit(1)####todo error throw out

        self.originDim = originDim
        self.lossDimEachSample = int(np.floor(originDim*lossRate))
        # print(self.lossDimEachSample,type(self.lossDimEachSample))
        self.lossSetValue = lossSetValue

    def lossSimulate(self, dataX):
        if len(dataX.shape) < 2:
            datarowdim = dataX.shape
        elif len(dataX.shape) > 2:
            print('Unable to process loss on matrix whose dim is above 3\n')
            exit(1)  ####todo error throw out
        else:
            sample = dataX.shape[0]
            datarowdim = dataX.shape[1]

        if datarowdim<self.originDim:
            print('Error in dataLossSimulator: dataDim is smaller than originDim\n')
            exit(1)  ####todo error throw out
        elif datarowdim>self.originDim:
            print('Warning in dataLossSimulator: dataDim is larger than originDim,'
                  'only former part dim of data exist loss\n')

        lossDataX = []
        for sample in dataX:
            lossDataSample = sample
            lossLocation = random.sample(range(self.originDim), self.lossDimEachSample)
            for loss in lossLocation:
                lossDataSample[loss] = self.lossSetValue

            lossDataX.append(lossDataSample)

        # print(lossLocation)
        # print(lossDataX)
        return np.array(lossDataX)


if __name__ == '__main__':
    lossTest = dataLossSimulator(4, 0.3)
    testData = myLoadData.loadIris().DataTrainX
    print(testData)
    lossdata = lossTest.lossSimulate(testData)
    print(lossdata)
