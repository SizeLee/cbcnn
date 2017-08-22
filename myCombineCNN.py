import numpy as np
import matplotlib.pyplot as plt
import json, threading
import myLoadData
from MyCombCNNPack import accuracyEvaluate, combineFeature, combineNumCalculate, costFunc, convLayer, fullConnect, \
    maxPoolingLayer, myException


class myCombineCNN:
    def __init__(self, data, combineNumConv1, convCoreNum1, combineNumPooling1):
        self.data = data
        self.combineNumConv1 = combineNumConv1
        self.combConvLayer1 = None
        self.convCoreNum1 = convCoreNum1
        self.convCoreList1 = list()
        self.convCoreOut1 = None

        self.poolingCoreList1 = list()
        self.poolingCoreOut1 = None
        self.combineNumPooling1 = combineNumPooling1
        self.combPoolingLayer1 = None

        self.allConnectData = None
        self.fullInputLayer = None

        self.midACData = None
        self.fullMidLayer = None

        self.predictResult = None

        self.poolingSFlist = None
        self.convSFlist = None

        self.trainInitializeFlag = False

        self.__trainingProgress = 0.

    def __modelExist(self):
        if self.combConvLayer1 is None or \
            not self.convCoreList1 or \
            not self.poolingCoreList1 or \
            self.combPoolingLayer1 is None or\
            self.fullInputLayer is None or\
            self.fullMidLayer is None:
            return False
        else:
            return True


    def saveModel(self, fname):
        if not self.__modelExist():
            return False
        model = {}
        model['convLayer'] = {}
        model['convLayer']['combConv'] = {}
        model['convLayer']['combConv']['all'] = self.combConvLayer1.getFeatureNum()
        model['convLayer']['combConv']['take'] = self.combConvLayer1.getCombineNum()
        model['convLayer']['convCore'] = {}
        model['convLayer']['convCore']['Num'] = self.convCoreNum1
        model['convLayer']['convCore']['weight'] = []

        for i in range(self.convCoreNum1):
            model['convLayer']['convCore']['weight'].append(self.convCoreList1[i].getWeight())

        model['poolingLayer'] = {}
        model['poolingLayer']['combPooling'] = {}
        # midCombNum = combineNumCalculate.combineNumCal(self.data.DataX.shape[1], self.combineNumConv1)
        model['poolingLayer']['combPooling']['all'] = self.combPoolingLayer1.getFeatureNum()
        model['poolingLayer']['combPooling']['take'] = self.combPoolingLayer1.getCombineNum()

        model['fullConnect'] = {}
        model['fullConnect']['inputLayer'] = {}
        model['fullConnect']['inputLayer']['inputDataShape'] = []
        model['fullConnect']['inputLayer']['inputDataShape'].append(10)###doesn't matter
        inputDataShape1 = combineNumCalculate.combineNumCal(self.combPoolingLayer1.getFeatureNum(),
                                                            self.combPoolingLayer1.getCombineNum()) * self.convCoreNum1
        model['fullConnect']['inputLayer']['inputDataShape'].append(inputDataShape1)
        model['fullConnect']['inputLayer']['weight'] = self.fullInputLayer.getWeight()

        model['fullConnect']['midLayer'] = {}
        model['fullConnect']['midLayer']['midInputDataShape'] = []
        model['fullConnect']['midLayer']['midInputDataShape'].append(10)##doesn't matter
        model['fullConnect']['midLayer']['midInputDataShape'].append(len(self.fullMidLayer.getWeight()))
        model['fullConnect']['midLayer']['yclassNum'] = self.data.DataTrainY.shape[1]
        model['fullConnect']['midLayer']['weight'] = self.fullMidLayer.getWeight()

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

        self.combConvLayer1 = combineFeature.combineFeature(model['convLayer']['combConv']['all'],
                                                            model['convLayer']['combConv']['take'])
        self.convCoreList1 = list()
        self.convCoreNum1 = model['convLayer']['convCore']['Num']
        for i in range(model['convLayer']['convCore']['Num']):

            convCoreTemp = convLayer.convLayerCore(1, 0.1)
            convCoreTemp.setWeight(model['convLayer']['convCore']['weight'][i])
            self.convCoreList1.append(convCoreTemp)

        self.combPoolingLayer1 = combineFeature.combineFeature(model['poolingLayer']['combPooling']['all'],
                                                               model['poolingLayer']['combPooling']['take'])
        self.poolingCoreList1 = list()
        for i in range(model['convLayer']['convCore']['Num']):
            poolingCoreTemp = maxPoolingLayer.maxPoolingLayerCore()
            self.poolingCoreList1.append(poolingCoreTemp)

        self.fullInputLayer = fullConnect.fullConnectInputLayer(model['fullConnect']['inputLayer']['inputDataShape'],
                                                                0.2)
        self.fullInputLayer.setWeight(model['fullConnect']['inputLayer']['weight'])

        self.fullMidLayer = fullConnect.fullConnectMidLayer(model['fullConnect']['midLayer']['midInputDataShape'],
                                                            model['fullConnect']['midLayer']['yclassNum'],
                                                            0.2)
        self.fullMidLayer.setWeight(model['fullConnect']['midLayer']['weight'])

        self.trainInitializeFlag = True

        # self.forwardPropagation(self.data.DataTestX)
        # print(self.predictResult)
        # print(costFunc.costCal(self.predictResult, self.data.DataTestY))
        # print(self.data.DataTestY)
        # print(accuracyEvaluate.classifyAccuracyRate(self.predictResult, self.data.DataTestY))
        return True


    def trainCNN(self, trainRound, trainRate, trainingContinueFlag, trainPicAccessLock):
        self.__trainingProgress = 0.
        self.combConvLayer1 = combineFeature.combineFeature(self.data.DataX.shape[1], self.combineNumConv1)
        combKindNumConv1 = combineNumCalculate.combineNumCal(self.data.DataX.shape[1], self.combineNumConv1)
        inputDataX = self.combConvLayer1.makeCombineData(self.data.DataTrainX)

        self.convCoreList1 = list()
        self.convCoreOut1 = list()
        for i in range(self.convCoreNum1):

            convCoreTemp = convLayer.convLayerCore(inputDataX.shape[2], trainRate)
            self.convCoreList1.append(convCoreTemp)
            self.convCoreOut1.append(convCoreTemp.calculate(inputDataX))

        self.combPoolingLayer1 = combineFeature.combineFeature(combKindNumConv1, self.combineNumPooling1)
        combKindNumPooling1 = combineNumCalculate.combineNumCal(combKindNumConv1, self.combineNumPooling1)

        self.poolingCoreList1 = list()
        self.poolingCoreOut1 = list()
        for i in range(self.convCoreNum1):

            inputPoolingData = self.combPoolingLayer1.makeCombineData(self.convCoreOut1[i])
            poolingCoreTemp = maxPoolingLayer.maxPoolingLayerCore()
            self.poolingCoreList1.append(poolingCoreTemp)
            self.poolingCoreOut1.append(poolingCoreTemp.calculate(inputPoolingData))

        for i in range(self.convCoreNum1):

            if self.allConnectData is None:
                self.allConnectData = self.poolingCoreOut1[i]
            else:
                self.allConnectData = np.hstack((self.allConnectData, self.poolingCoreOut1[i]))

        # print(self.allConnectData)
        # print(self.allConnectData.shape)

        self.fullInputLayer = fullConnect.fullConnectInputLayer(self.allConnectData.shape, trainRate)
        self.midACData = self.fullInputLayer.calculate(self.allConnectData)
        self.fullMidLayer = fullConnect.fullConnectMidLayer(self.midACData.shape, self.data.DataTrainY.shape[1], trainRate)
        self.predictResult = self.fullMidLayer.calculate(self.midACData)

        # print(self.predictResult)
        # print(self.data.DataTrainY)
        # print(self.predictResult.shape)
        # print(self.data.DataTrainY.shape)

        # todo BP process
        ####### full connect BP
        formerLayerSF = self.fullMidLayer.BP(self.data.DataTrainY)
        # print(formerLayerSF)
        # print(formerLayerSF.shape)
        formerLayerSF = self.fullInputLayer.BP(formerLayerSF)
        # print(formerLayerSF)

        ####### max pooling layer BP
        splitStep = int(formerLayerSF.shape[1] / self.convCoreNum1)

        self.poolingSFlist = list()
        for i in range(self.convCoreNum1):
            SFtemp = formerLayerSF[:, i * splitStep : (i + 1) * splitStep].copy()
            # print(SFtemp.shape)
            self.poolingSFlist.append(SFtemp)

        formerLayerSF = list()
        for i in range(self.convCoreNum1):
            formerLayerSF.append(self.poolingCoreList1[i].BP(self.poolingSFlist[i]))

        # print(formerLayerSF[0])

        ######################    combine feature BP

        self.convSFlist = list()
        for i in range(self.convCoreNum1):
            self.convSFlist.append(self.combPoolingLayer1.BP(formerLayerSF[i]))

        # print(formerLayerSF)
        # print(formerLayerSF[0].shape)
        # print(len(formerLayerSF))

        #####################   conv layer BP
        for i in range(self.convCoreNum1):
            self.convCoreList1[i].BP(self.convSFlist[i])

        self.trainInitializeFlag = True

        trainCost = costFunc.costCal(self.predictResult, self.data.DataTrainY)
        lastTrainCost = trainCost
        trainCostList = []
        trainCostList.append(trainCost)
        trainTimeList = [0]
        ###################### start train in round
        for trainTime in range(trainRound - 1):
            if not trainingContinueFlag[0]:
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

            # if trainCost > lastTrainCost:
            #     trainRate = trainRate / 2
            #     for i in range(len(self.convCoreList1)):
            #         self.convCoreList1[i].setTrainRate(trainRate)
            #
            #     self.fullInputLayer.setTrainRate(trainRate)
            #     self.fullMidLayer.setTrainRate(trainRate)
            #     print(trainRate)
            #
            # lastTrainCost = trainCost

            trainCostList.append(trainCost)
            trainTimeList.append(trainTime + 1)
            self.__trainingProgress = (trainTime + 2) / float(trainRound)
            #     progressBar.setValue(np.ceil((trainTime + 1) / float(trainRound) * 100))
            if (trainTime + 1) % 5 == 0:
                plt.figure(figsize=(8,5))
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


    def forwardPropagation(self, inputDataX = None):

        if self.trainInitializeFlag == False:
            print("Can\'t forwardPropagation CNN before train initialize\n")
            exit(1)#todo throw out error

        # self.combConvLayer1 = combineFeature.combineFeature(self.data.DataX.shape[1], self.combineNumConv1)
        # combKindNumConv1 = combineNumCalculate.combineNumCal(self.data.DataX.shape[1], self.combineNumConv1)
        # inputDataX = self.combConvLayer1.makeCombineData(self.data.DataTrainX)
        if inputDataX is None:
            inputDataX = self.combConvLayer1.makeCombineData(self.data.DataTrainX)
        else:
            inputDataX = self.combConvLayer1.makeCombineData(inputDataX)

        self.convCoreOut1 = list()
        for i in range(self.convCoreNum1):

            self.convCoreOut1.append(self.convCoreList1[i].calculate(inputDataX))

        # self.combPoolingLayer1 = combineFeature.combineFeature(combKindNumConv1, self.combineNumPooling1)
        # combKindNumPooling1 = combineNumCalculate.combineNumCal(combKindNumConv1, self.combineNumPooling1)

        self.poolingCoreOut1 = list()
        for i in range(self.convCoreNum1):
            inputPoolingData = self.combPoolingLayer1.makeCombineData(self.convCoreOut1[i])
            self.poolingCoreOut1.append(self.poolingCoreList1[i].calculate(inputPoolingData))

        self.allConnectData = None
        for i in range(self.convCoreNum1):

            if self.allConnectData is None:
                self.allConnectData = self.poolingCoreOut1[i]
            else:
                self.allConnectData = np.hstack((self.allConnectData, self.poolingCoreOut1[i]))

        # print(self.allConnectData)
        # print(self.allConnectData.shape)

        # self.fullInputLayer = fullConnect.fullConnectInputLayer(self.allConnectData, trainRate)
        self.midACData = self.fullInputLayer.calculate(self.allConnectData)
        # self.fullMidLayer = fullConnect.fullConnectMidLayer(self.midACData, self.data.DataTrainY, trainRate)
        self.predictResult = self.fullMidLayer.calculate(self.midACData)

        # print(self.predictResult)

    def backPropagation(self):
        ####### full connect BP
        formerLayerSF = self.fullMidLayer.BP() ###existed ylabel , no need for pass ylabel
        # print(formerLayerSF)
        # print(formerLayerSF.shape)
        formerLayerSF = self.fullInputLayer.BP(formerLayerSF)
        # print(formerLayerSF)

        ####### max pooling layer BP
        splitStep = int(formerLayerSF.shape[1] / self.convCoreNum1)

        self.poolingSFlist = list()
        for i in range(self.convCoreNum1):
            SFtemp = formerLayerSF[:, i * splitStep: (i + 1) * splitStep].copy()
            # print(SFtemp.shape)
            self.poolingSFlist.append(SFtemp)

        formerLayerSF = list()
        for i in range(self.convCoreNum1):
            formerLayerSF.append(self.poolingCoreList1[i].BP(self.poolingSFlist[i]))

        # print(formerLayerSF[0])

        ######################    combine feature BP

        self.convSFlist = list()
        for i in range(self.convCoreNum1):
            self.convSFlist.append(self.combPoolingLayer1.BP(formerLayerSF[i]))

        # print(formerLayerSF)
        # print(formerLayerSF[0].shape)
        # print(len(formerLayerSF))

        #####################   conv layer BP
        for i in range(self.convCoreNum1):
            self.convCoreList1[i].BP(self.convSFlist[i])

    def getTrainingProgress(self):
        return self.__trainingProgress

    def getPredictResult(self):
        return self.predictResult.round(decimals=3)

    def getAccuratePredictResult(self):
        return self.predictResult.copy()


    def runCNN(self, setChoose = 'Train', data = None):
        if not self.__modelExist():
            ##no model exist in this instance
            raise myException.ModelExistException

        if data is not None:
            if isinstance(data, myLoadData.loadData):
                if data.DataX.shape[1] != self.combConvLayer1.getFeatureNum():
                    raise myException.DataModelMatchException

                self.data = data
            else:
                raise myException.DataValidFormatException

        else:
            if self.data is None:
                raise myException.DataExistException

            if isinstance(self.data, myLoadData.loadData):
                raise myException.DataValidFormatException

            if self.data.DataX.shape[1] != self.combConvLayer1.getFeatureNum():
                raise myException.DataModelMatchException

        if setChoose == 'Train':
            self.forwardPropagation()

        elif setChoose == 'Test':
            self.forwardPropagation(self.data.DataTestX)

        elif setChoose == 'Validation':
            self.forwardPropagation(self.data.DataValX)

if __name__ == '__main__':
    # irisDATA = myLoadData.loadIris(0.3, -1)
    # mcnn = myCombineCNN(irisDATA, 2, 5, 4)
    # mcnn.trainCNN(1600,0.2, [True])

    irisDATA = myLoadData.loadData('..\\iris.txt', 0.3, -1)
    mcnn = myCombineCNN(irisDATA, 2, 5, 4)
    mcnn.trainCNN(1600,0.2, [True], threading.Lock())





# self.data = None
# if dataName == 'Iris':
#     self.data = myLoadData.loadIris()
# else:
#     print('No such data file, Waiting for new data set\n')
#     exit(1) # todo throw error or set new data






