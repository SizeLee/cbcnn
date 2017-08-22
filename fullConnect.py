import numpy as np

from MyCombCNNPack import activationFunction


class fullConnectInputLayer:

    def __init__(self, inputDataShape, trainRate, LoutinRate = 0.5):
        ''' inputDataX should be two dim, transpose 3dim data(sample, pooling data, core num)
                                     into two dim(sample, pooling data*core num) '''
        if len(inputDataShape) != 2:
            print("Error in fullConectLayer: inputDataX isn't 2 dim\n")
            exit(1) #todo throw error

        self.__inputDataX = None
        self.sampleNum = inputDataShape[0]
        self.__L_in = inputDataShape[1] + 1
        self.__LoutinRate = LoutinRate
        self.__L_out = int(np.floor(self.__LoutinRate * self.__L_in))
        self.epsilon = np.sqrt(6)/(np.sqrt(self.__L_in) + np.sqrt(self.__L_out))
        self.__w = np.random.rand(self.__L_in, self.__L_out) * 2 * self.epsilon - self.epsilon
        # self.__inputDataX = np.hstack((np.ones((self.sampleNum, 1)), inputDataX))
        # print(self.__inputDataX)
        self.__outputDataX = None
        self.__outputDataAct = None
        self.__trainRate = trainRate

    def getWeight(self):
        return self.__w.tolist()

    def setWeight(self, wList):
        self.__w = np.array(wList)

    def setTrainRate(self, rate):
        self.__trainRate = rate

    def calculate(self, newInputDataX = None):

        if newInputDataX is not None:

            if self.__inputDataX is not None and self.__inputDataX[:, 1:].shape[1] != newInputDataX.shape[1]:
                print('Error in full connect input layer: new data is with wrong size\n')
                exit(1)  # todo throw out error
            else:
                self.sampleNum = newInputDataX.shape[0]
                self.__inputDataX = np.hstack((np.ones((self.sampleNum, 1)), newInputDataX))    ##########iterate new data into mid layer

        self.__outputDataX = np.dot(self.__inputDataX, self.__w)
        self.__outputDataAct = activationFunction.sigmoid(self.__outputDataX)

        return self.__outputDataAct.copy()

    #todo def BP function

    def BP(self, sensitivityFactor):

        if self.__inputDataX is None:
            print('Error in full connect layer BP: no exist input data in full connect layer for BP\n')
            exit(1)  # todo throw out error

        delt = sensitivityFactor * activationFunction.sigmoidGradient(self.__outputDataX)
        wGradient = 1/self.sampleNum * np.dot(self.__inputDataX.T, delt)

        sensitivityFactorFormerLayer = np.dot(delt, self.__w[1:, :].T)
        self.__w = self.__w - self.__trainRate * wGradient

        return sensitivityFactorFormerLayer




class fullConnectMidLayer:

    def __init__(self, midInputDataShape, yClassNum, trainRate):
        '''midInputDataX is sample * midfeature, y is sample * yLabel'''

        if len(midInputDataShape) != 2:
            print("Error in fullConectLayer: inputDataX isn't 2 dim")
            exit(1) #todo throw error

        self.__midInputDataX = None
        self.sampleNum = midInputDataShape[0]
        self.__L_in = midInputDataShape[1] + 1
        self.__L_out = yClassNum
        self.epsilon = np.sqrt(6) / (np.sqrt(self.__L_in) + np.sqrt(self.__L_out))
        self.__w = np.random.rand(self.__L_in, self.__L_out) * 2 * self.epsilon - self.epsilon
        # self.__midInputDataX = np.hstack((np.ones((self.sampleNum, 1)), midInputDataX))
        # print(self.__midInputDataX)
        self.__outputY = None
        self.__outputYAct = None
        self.__yLabel = None
        self.__trainRate = trainRate

    def getWeight(self):
        return self.__w.tolist()

    def setWeight(self, wList):
        self.__w = np.array(wList)

    def setTrainRate(self, rate):
        self.__trainRate = rate

    def calculate(self, newInputDataX = None):

        if newInputDataX is not None:

            if self.__midInputDataX is not None and self.__midInputDataX[:, 1:].shape[1] != newInputDataX.shape[1]:
                print('Error in full connect mid layer: new data is with wrong size\n')
                exit(1)  # todo throw out error
            else:
                self.sampleNum = newInputDataX.shape[0]
                self.__midInputDataX = np.hstack((np.ones((self.sampleNum, 1)), newInputDataX))    ##########iterate new data into mid layer

        self.__outputY = np.dot(self.__midInputDataX, self.__w)
        self.__outputYAct = activationFunction.sigmoid(self.__outputY)

        return self.__outputYAct.copy()

    #todo def BP function
    def BP(self, ylabel = None):
        if ylabel is not None:
            if self.__yLabel is not None and self.__yLabel.shape[1] != ylabel.shape[1]:
                print("Error in fullConnect mid layer BP: wrong class Num of y-label data passed into full connect mid layer\n")
                exit(1) #todo throw out error
            else:
                self.__yLabel = ylabel

        if self.__yLabel is None:
            print("Error in fullConnect mid layer BP: no exist y-label data in full connect mid layer for BP\n")
            exit(1) #todo throw out error

        delt = (self.__outputYAct - self.__yLabel) * activationFunction.sigmoidGradient(self.__outputY)
        wGradient = 1/self.sampleNum * np.dot(self.__midInputDataX.T, delt)
        sensitivityFactorFormerLayer = np.dot(delt, self.__w[1:,:].T)

        self.__w = self.__w - self.__trainRate * wGradient

        return sensitivityFactorFormerLayer



