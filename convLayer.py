import numpy as np
import myLoadData
import activationFunction, combineFeature


class convLayerCore:
    def __init__(self, wLength, trainRate):
        self.__w = None
        self.__inputDataX = None
        self.__outputDataX = None
        self.__outputDataAct = None
        # sampleNum = InputDataX.shape[0]
        # combineNum = InputDataX.shape[1]
        # combineFeatureNum = InputDataX.shape[2]
        # if wLength!=combineFeatureNum:
        #     print('Error in convlayer.Wrong conv core length\n')
        #     exit(1)#todo throw error

        # self.__inputDataX = InputDataX.copy()

        self.__w = 0.24 * np.random.rand(wLength, 1) - 0.12
        # print(self.__w)
        self.__leakyRate = 0.5
        self.__trainRate = trainRate

    def getWeight(self):
        return self.__w.tolist()

    def setWeight(self, wList):
        self.__w = np.array(wList)

    def setTrainRate(self, rate):
        self.__trainRate = rate

    def calculate(self, newInputDataX = None):
        if newInputDataX is not None:
            if self.__inputDataX is not None and self.__inputDataX.shape[1] != newInputDataX.shape[1]:
                print('Error in conv layer: new data is with wrong size\n')
                exit(1) #todo throw out error
            else:
                self.__inputDataX = newInputDataX.copy()    ####iterate newdata into convLayer

        if self.__inputDataX is None:
            print('Error in conv layer: no exist data in conv layer core for calculation\n')
            exit(1) #todo throw out error

        self.__outputDataX = np.dot(self.__inputDataX, self.__w)
        # print(np.dot(self.__inputDataX, self.__w).shape)

        self.__outputDataAct = activationFunction.leakyReLU(self.__outputDataX, self.__leakyRate)

        return self.__outputDataAct.reshape(self.__outputDataAct.shape[0], self.__outputDataAct.shape[1])

    #todo def BP function

    def BP(self, sensitivityFactor):
        sensitivityFactor = sensitivityFactor.reshape((sensitivityFactor.shape[0], sensitivityFactor.shape[1], 1))
        if self.__outputDataX is None:
            print('Error in conv layer BP: no exist output data in conv layer core for BP\n')
            exit(1) #todo throw out error

        if sensitivityFactor.shape != self.__outputDataX.shape:
            print('Error in convLayer BP: input wrong sensitivity factor\n')
            exit(1) #todo throw out

        if self.__inputDataX is None:
            print('Error in conv layer BP: no exist input data in conv layer core for calculation\n')
            exit(1) #todo throw out error

        sampleNum = sensitivityFactor.shape[0]

        delt = sensitivityFactor * activationFunction.leakyReLUGradient(self.__outputDataX, self.__leakyRate)

        wGradient = np.zeros(self.__w.shape)

        for i in range(sampleNum):
            wGradient += np.dot(self.__inputDataX[i, :, :].T, sensitivityFactor[i, :, :])

        wGradient *= (1/float(sampleNum))

        formerLayerSF = list()
        for i in range(sampleNum):
            formerLayerSF.append(np.dot(sensitivityFactor[i, :, :], self.__w.T))

        formerLayerSF = np.array(formerLayerSF)

        self.__w = self.__w - self.__trainRate * wGradient

        # print(formerLayerSF)
        # print(self.__w)
        # print(formerLayerSF.shape)
        # print(self.__w.shape)

        return formerLayerSF


class MultiConv:
    def __init__(self, convCoreNum, combLength, trainRate):
        self.__wLength = combLength+1
        self.__w = None
        self.__inputDataX = None
        self.__outputDataX = None
        self.__outputDataAct = None
        # sampleNum = InputDataX.shape[0]
        # combineNum = InputDataX.shape[1]
        # combineFeatureNum = InputDataX.shape[2]
        # if wLength!=combineFeatureNum:
        #     print('Error in convlayer.Wrong conv core length\n')
        #     exit(1)#todo throw error

        # self.__inputDataX = InputDataX.copy()

        self.__w = 0.24 * np.random.rand(convCoreNum, self.__wLength) - 0.12
        # print(self.__w)
        self.__leakyRate = 0.5
        self.__trainRate = trainRate

    def getWeight(self):
        return self.__w.tolist()

    def setWeight(self, wList):
        self.__w = np.array(wList)

    def setTrainRate(self, rate):
        self.__trainRate = rate

    def calculate(self, newInputDataX = None):
        if newInputDataX is not None:
            if self.__inputDataX is not None and self.__inputDataX.shape[1] != newInputDataX.shape[1]:
                print('Error in conv layer: new data is with wrong size\n')
                exit(1) #todo throw out error
            else:
                self.__inputDataX = newInputDataX    ####iterate newdata into convLayer

        if self.__inputDataX is None:
            print('Error in conv layer: no exist data in conv layer core for calculation\n')
            exit(1) #todo throw out error

        sample = self.__inputDataX.shape[0]
        combnum = self.__inputDataX.shape[1]
        wlength = self.__inputDataX.shape[2]

        datatemp = self.__inputDataX.reshape(sample*combnum, wlength)
        datatemp = np.hstack((np.ones((sample*combnum, 1)), datatemp))
        wlength += 1
        self.__inputDataX = datatemp.reshape(sample, combnum, wlength)

        self.__outputDataX = np.sum(self.__inputDataX * self.__w, axis=2)
        # print(np.dot(self.__inputDataX, self.__w).shape)

        self.__outputDataAct = activationFunction.leakyReLU(self.__outputDataX, self.__leakyRate)

        return self.__outputDataAct

    #todo def BP function

    def BP(self, sensitivityFactor):
        delt = sensitivityFactor * activationFunction.leakyReLUGradient(self.__outputDataX, self.__leakyRate)
        delt = delt.reshape((delt.shape[0], delt.shape[1], 1))
        sampleNum = delt.shape[0]
        # if self.__outputDataX is None:
        #     print('Error in conv layer BP: no exist output data in conv layer core for BP\n')
        #     exit(1) #todo throw out error
        #
        # if sensitivityFactor.shape != self.__outputDataX.shape:
        #     print('Error in convLayer BP: input wrong sensitivity factor\n')
        #     exit(1) #todo throw out
        #
        # if self.__inputDataX is None:
        #     print('Error in conv layer BP: no exist input data in conv layer core for calculation\n')
        #     exit(1) #todo throw out error

        wGradient = 1/float(sampleNum) * np.sum(self.__inputDataX * delt, axis=0)

        # sampleNum = sensitivityFactor.shape[0]

        # delt = sensitivityFactor * activationFunction.leakyReLUGradient(self.__outputDataX, self.__leakyRate)

        # wGradient = np.zeros(self.__w.shape)
        #
        # for i in range(sampleNum):
        #     wGradient += np.dot(self.__inputDataX[i, :, :].T, sensitivityFactor[i, :, :])
        #
        # wGradient *= (1/float(sampleNum))

        # formerLayerSF = list()
        # for i in range(sampleNum):
        #     formerLayerSF.append(np.dot(sensitivityFactor[i, :, :], self.__w.T))
        #
        # formerLayerSF = np.array(formerLayerSF)

        formerLayerSF = delt * self.__w.reshape((1, self.shape[0], self.shape[1]))

        self.__w = self.__w - self.__trainRate * wGradient

        # print(formerLayerSF)
        # print(self.__w)
        # print(formerLayerSF.shape)
        # print(self.__w.shape)

        return formerLayerSF




if __name__ == '__main__':
    irisData = myLoadData.loadIris()
    comb = combineFeature.combineFeature(4,2)
    inputDataX = comb.makeCombineData(irisData.DataTestX)
    testConvCore = convLayerCore(inputDataX, inputDataX.shape[2])
    convOut = testConvCore.calculate()
    print(convOut)
    print(inputDataX)