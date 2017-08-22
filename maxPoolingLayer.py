import numpy as np
import myLoadData
from MyCombCNNPack import combineFeature, convLayer

class maxPoolingLayerCore:

    def __init__(self):

        self.__inputDataX = None
        self.__poolingSize = None
        self.__poolingPosition = None
        self.__outputDataX = None

    def calculate(self, newInputDataX = None):
        if newInputDataX is not None:
            if self.__inputDataX is not None and self.__inputDataX.shape[1] != newInputDataX.shape[1]:
                print('Error in pooling layer: new data is with wrong size\n')
                exit(1)  # todo throw out error
            else:
                self.__inputDataX = newInputDataX.copy()  ####iterate newdata into max pooling layer

        if self.__inputDataX is None:
            print('Error in maxPooling layer: no exist data in maxPooling layer core for calculation\n')
            exit(1) #todo throw out error

        self.__poolingSize = self.__inputDataX.shape[2]
        self.__outputDataX = np.max(self.__inputDataX, axis=2) ##obtain max of each row in each sample, it's max pooling
        self.__poolingPosition = np.argmax(self.__inputDataX, axis=2) ##obtain taking value's position, for latter BP process
        # print(self.__outputDataX)
        # print(self.__poolingPosition)

        return self.__outputDataX.copy()

    def outputCalculateResult(self):
        print(self.__outputDataX)


    #todo def BP function
    def BP(self, sensitivityFactor):

        if self.__inputDataX is None:
            print('Error in maxPooling layer BP: no exist input data in maxPooling layer core for BP\n')
            exit(1) #todo throw out error

        if sensitivityFactor.shape[0] != self.__inputDataX.shape[0] \
                or sensitivityFactor.shape[1] != self.__inputDataX.shape[1]:
            print('Error in maxpooling BP: input wrong sensitivity factor\n')
            exit(1) #todo throw out error

        sampleNum = sensitivityFactor.shape[0]
        combNum = sensitivityFactor.shape[1]
        featureNum = self.__inputDataX.shape[2]
        # print(sampleNum,combNum,featureNum)

        formerSF = np.zeros((sampleNum, combNum, featureNum))

        for i in range(sampleNum):
            for j in range(combNum):
                formerSF[i, j, self.__poolingPosition[i, j]] = sensitivityFactor[i, j]

        # print(formerSF)
        # print(self.__poolingPosition)
        return formerSF

if __name__ == '__main__':
    irisData = myLoadData.loadIris()
    comb = combineFeature.combineFeature(4,2)
    inputDataX = comb.makeCombineData(irisData.DataTestX)
    testConvCore = convLayer.convLayerCore(inputDataX, inputDataX.shape[2])
    convOut = testConvCore.calculate()
    # print(convOut)
    # print(convOut.shape)
    comb2 = combineFeature.combineFeature(6,4)
    # comb2.outputCombineMap()
    inputPoolingDataX = comb2.makeCombineData(convOut)
    # print(inputPoolingDataX)
    testPoolingCore = maxPoolingLayerCore(inputPoolingDataX)
    result = testPoolingCore.calculate()
    # print(result)
    result[:,:]=1
    testPoolingCore.outputCalculateResult()

