import numpy as np
import matplotlib.pyplot as plt
import fullConnect, costFunc, accuracyEvaluate, myException
import threading, json
import myLoadData

class traditionalNN:
    def __init__(self, data):
        self.data = data
        self.inputLayer = None
        self.midData = None

        self.midLayer = None
        self.predictResult = None

        self.__trainingProgress = 0.

    def getTrainingProgress(self):
        return self.__trainingProgress

    def getPredictResult(self):
        return self.predictResult.round(decimals=3)

    def getAccuratePredictResult(self):
        return self.predictResult.copy()

    def train(self, trainRound, trainRate, LoutinRate, trainContinueFlag, trainPicAccessLock):

        self.inputLayer = fullConnect.fullConnectInputLayer(self.data.DataTrainX.shape, trainRate, LoutinRate)
        self.midData = self.inputLayer.calculate(self.data.DataTrainX)
        self.midLayer = fullConnect.fullConnectMidLayer(self.midData.shape, self.data.DataTrainY.shape[1], trainRate)
        self.predictResult = self.midLayer.calculate(self.midData)

        formerSF = self.midLayer.BP(self.data.DataTrainY)
        formerSF = self.inputLayer.BP(formerSF)

        trainCost = costFunc.costCal(self.predictResult, self.data.DataTrainY)
        trainCostList = []
        trainCostList.append(trainCost)
        trainTimeList = [0]
        ###################### start train in round
        for trainTime in range(trainRound - 1):
            if not trainContinueFlag[0]:
                break
            # print('1')
            self.forwardPropagation()
            # print('mid')
            self.backPropagation()
            # print('2')
            trainCost = costFunc.costCal(self.predictResult, self.data.DataTrainY)
            # self.forwardPropagation(self.combConvLayer1.makeCombineData(self.data.DataValX))
            # valCost = costFunc.costCal(self.predictResult, self.data.DataValY)
            # print(trainCost, valCost)
            # print(trainCost)
            trainCostList.append(trainCost)
            trainTimeList.append(trainTime + 1)
            self.__trainingProgress = (trainTime + 1) / float(trainRound)
            #     progressBar.setValue(np.ceil((trainTime + 1) / float(trainRound) * 100))
            if (trainTime + 1) % 5 == 0:
                plt.figure(figsize=(8, 5))
                plt.plot(trainTimeList, trainCostList, 'b-')
                plt.xlabel('Training times')
                plt.ylabel('Training cost')
                plt.xlim(0, trainRound + 2)
                plt.ylim(0, 1.6 * trainCostList[0])
                trainPicAccessLock.acquire()
                plt.savefig('TrainingCost.png')
                trainPicAccessLock.release()
                plt.close()

        print(accuracyEvaluate.classifyAccuracyRate(self.predictResult, self.data.DataTrainY))

        self.forwardPropagation(self.data.DataTestX)
        print(self.predictResult)
        print(costFunc.costCal(self.predictResult, self.data.DataTestY))
        print(self.data.DataTestY)
        print(accuracyEvaluate.classifyAccuracyRate(self.predictResult, self.data.DataTestY))


    def runTraNN(self, setChoose = 'Train', data = None):
        if not self.__modelExist():
            ##no model exist in this instance
            raise myException.ModelExistException

        if data is not None:
            if isinstance(data, myLoadData.loadData):
                if data.DataX.shape[1] != len(self.inputLayer.getWeight()) - 1:
                    raise myException.DataModelMatchException

                self.data = data
            else:
                raise myException.DataValidFormatException

        else:
            if self.data is None:
                raise myException.DataExistException

            if isinstance(self.data, myLoadData.loadData):
                raise myException.DataValidFormatException

            if self.data.DataX.shape[1] != len(self.inputLayer.getWeight()):
                raise myException.DataModelMatchException


        if setChoose == 'Train':
            self.forwardPropagation()

        elif setChoose == 'Test':
            self.forwardPropagation(self.data.DataTestX)

        elif setChoose == 'Validation':
            self.forwardPropagation(self.data.DataValX)


    def forwardPropagation(self, inputDataX = None):
        if inputDataX is None:
            inputDataX = self.data.DataTrainX

        self.midData = self.inputLayer.calculate(inputDataX)
        self.predictResult = self.midLayer.calculate(self.midData)


    def backPropagation(self):
        formerSF = self.midLayer.BP(self.data.DataTrainY)
        formerSF = self.inputLayer.BP(formerSF)


    def __modelExist(self):
        if self.inputLayer is None or\
           self.midLayer is None:
            return False
        else:
            return True

    def saveModel(self, fname):
        if not self.__modelExist():
            return False
        model = {}

        model['fullConnect'] = {}
        model['fullConnect']['inputLayer'] = {}
        model['fullConnect']['inputLayer']['inputDataShape'] = []
        model['fullConnect']['inputLayer']['inputDataShape'].append(10) ## doesn't matter
        inputDataShape1 = len(self.inputLayer.getWeight())
        model['fullConnect']['inputLayer']['inputDataShape'].append(inputDataShape1)
        model['fullConnect']['inputLayer']['weight'] = self.inputLayer.getWeight()

        model['fullConnect']['midLayer'] = {}
        model['fullConnect']['midLayer']['midInputDataShape'] = []
        model['fullConnect']['midLayer']['midInputDataShape'].append(10)###doesn't matter
        model['fullConnect']['midLayer']['midInputDataShape'].append(len(self.midLayer.getWeight()))##Only about initialization, doesn't matter
        model['fullConnect']['midLayer']['yclassNum'] = len(self.midLayer.getWeight()[0])
        model['fullConnect']['midLayer']['weight'] = self.midLayer.getWeight()

        try:
            fp = open(fname, 'w')
            json.dump(model, fp)
            fp.close()

        except FileNotFoundError:
            return False

        return True


    def setModel(self, fname):
        try:
            fp = open(fname, 'r')
            model = json.load(fp)
            fp.close()

        except FileNotFoundError:
            return False

        self.inputLayer = fullConnect.fullConnectInputLayer(model['fullConnect']['inputLayer']['inputDataShape'],
                                                            0.2)
        self.inputLayer.setWeight(model['fullConnect']['inputLayer']['weight'])

        self.midLayer = fullConnect.fullConnectMidLayer(model['fullConnect']['midLayer']['midInputDataShape'],
                                                        model['fullConnect']['midLayer']['yclassNum'],
                                                        0.2)
        self.midLayer.setWeight(model['fullConnect']['midLayer']['weight'])

        # self.forwardPropagation(self.data.DataTestX)
        # print(self.predictResult)
        # print(costFunc.costCal(self.predictResult, self.data.DataTestY))
        # print(self.data.DataTestY)
        # print(accuracyEvaluate.classifyAccuracyRate(self.predictResult, self.data.DataTestY))
        return True

if __name__ == '__main__':
    irisDATA = myLoadData.loadData('..\\iris.txt')
    traNN = traditionalNN(irisDATA)
    traNN.train(500, 0.9, 2, [True], threading.Lock())