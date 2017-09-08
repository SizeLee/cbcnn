import numpy as np
import random
import dataLossSimulator
import copy
class loadIris:
    # DataX = None
    # DataTrainX = None
    # DataValX = None
    # DataTestX = None
    #
    # DataY = None
    # DataTrainY = None
    # DataTestY = None
    # DataValY = None
    #
    # sampleList = None
    # sampleX = None
    # sampleY = None

    dataName = 'Iris'

    def __init__(self, lossRate = 0., setlossValue = 0.):
        fp = open('iris.txt','r') #todo change the way open file

        self.DataX = None
        self.DataTrainX = None
        self.DataTrainXLoss = None
        self.DataValX = None
        self.DataValXLoss = None
        self.DataTestX = None
        self.DataTestXLoss = None

        self.DataY = None
        self.DataTrainY = None
        self.DataTestY = None
        self.DataValY = None

        self.sampleList = None
        self.sampleX = None
        self.sampleY = None

        self.lossRate = lossRate
        self.setLossValue = setlossValue

        sampleCount = 0
        self.sampleList = []
        for line in fp:
            sampleCount += 1
            line = line.rstrip('\n')
            dataStrList = line.split(',')
            linetemp = []
            for data in dataStrList:
                if data == 'Iris-setosa':
                    linetemp.append([1,0,0])
                elif data == 'Iris-versicolor':
                    linetemp.append([0,1,0])
                elif data == 'Iris-virginica':
                    linetemp.append([0,0,1])
                elif linetemp == []:
                    linetemp.append([float(data)])
                else:
                    linetemp[0].append(float(data))
            # print(linetemp)
            self.sampleList.append(linetemp)

        # print(self.sampleList)
        # print(sampleCount)
        ##load into memory
        orderList = range(sampleCount)
        randomChange = random.sample(orderList,sampleCount)
        # print(randomChange)
        # randomChange.sort()
        # print(randomChange)
        sampleTemp = copy.deepcopy(self.sampleList)
        for i in orderList:
            self.sampleList[i] = sampleTemp[randomChange[i]]

        # print(sampleList)

        self.sampleX = []
        self.sampleY = []

        for line in self.sampleList:
            self.sampleX.append(line[0])
            self.sampleY.append(line[1])

        self.DataX = np.array(self.sampleX)
        self.DataY = np.array(self.sampleY,dtype=float)

        # print(self.IrisDataX)
        # print(self.IrisDataY)

        self.DataTrainX = self.DataX[:int(0.6 * sampleCount), :]
        self.DataTrainY = self.DataY[:int(0.6 * sampleCount), :]

        self.DataValX = self.DataX[int(0.6 * sampleCount):int(0.8 * sampleCount), :]
        self.DataValY = self.DataY[int(0.6 * sampleCount):int(0.8 * sampleCount), :]

        self.DataTestX = self.DataX[int(0.8 * sampleCount):, :]
        self.DataTestY = self.DataY[int(0.8 * sampleCount):, :]

        fp.close()

        # print(self.DataTrainX.shape)
        # print(self.DataTrainY.shape)
        #
        # print(self.DataValX.shape)
        # print(self.DataValY.shape)
        #
        # print(self.DataTestX.shape)
        # print(self.DataTestY.shape)

        self.lossSimulator = dataLossSimulator.dataLossSimulator(self.DataX.shape[1], self.lossRate, self.setLossValue)

        self.DataTrainXLoss = self.lossSimulator.lossSimulate(self.DataTrainX)
        self.DataValXLoss = self.lossSimulator.lossSimulate(self.DataValX)
        self.DataTestXLoss = self.lossSimulator.lossSimulate(self.DataTestX)

        self.DataTrainX = self.DataTrainXLoss
        self.DataValX = self.DataValXLoss
        self.DataTestX = self.DataTestXLoss


class loadData:

    def __init__(self, filedirectory, lossRate = 0., setlossValue = 0.):
        try:
            fp = open(filedirectory,'r')
        except Exception as e:
            raise e

        self.DataX = None
        self.DataTrainX = None
        self.DataTrainXLoss = None
        self.DataValX = None
        self.DataValXLoss = None
        self.DataTestX = None
        self.DataTestXLoss = None

        self.DataY = None
        self.DataTrainY = None
        self.DataTestY = None
        self.DataValY = None

        self.sampleList = None
        self.sampleX = None
        self.sampleY = None

        self.lossRate = lossRate
        self.setLossValue = setlossValue

        sampleCount = 0
        self.sampleList = []

        self.yClassDic = dict()
        self.yClassNum = 0
        for line in fp:
            line = line.rstrip('\n')
            dataStrList = line.split(',')
            if dataStrList[-1] not in self.yClassDic:
                self.yClassDic[dataStrList[-1]] = self.yClassNum
                self.yClassNum = self.yClassNum + 1

        # print(self.yClassDic)
        # print(len(self.yClassDic))
        ylabelVecTemplate = []
        for i in range(len(self.yClassDic)):
            ylabelVecTemplate.append(0)

        # print(ylabelVecTemplate)

        fp.seek(0, 0)
        for line in fp:
            sampleCount += 1
            line = line.rstrip('\n')
            dataStrList = line.split(',')
            linetemp = []
            for data in dataStrList:
                if data in self.yClassDic:
                    ylabelVec = copy.deepcopy(ylabelVecTemplate)
                    ylabelVec[self.yClassDic[data]] = 1
                    # print(ylabelVec)
                    linetemp.append(ylabelVec)
                elif linetemp == []:
                    linetemp.append([float(data)])
                else:
                    linetemp[0].append(float(data))
            # print(linetemp)
            self.sampleList.append(linetemp)

        # print(self.sampleList)
        # print(sampleCount)
        ##load into memory
        orderList = range(sampleCount)
        randomChange = random.sample(orderList,sampleCount)
        # print(randomChange)
        # randomChange.sort()
        # print(randomChange)
        sampleTemp = copy.deepcopy(self.sampleList)
        for i in orderList:
            self.sampleList[i] = sampleTemp[randomChange[i]]

        # print(sampleList)

        self.sampleX = []
        self.sampleY = []

        for line in self.sampleList:
            self.sampleX.append(line[0])
            self.sampleY.append(line[1])

        self.DataX = np.array(self.sampleX)
        self.DataY = np.array(self.sampleY,dtype=float)

        # print(self.IrisDataX)
        # print(self.IrisDataY)

        # self.DataTrainX = self.DataX[:int(0.6 * sampleCount), :]
        # self.DataTrainY = self.DataY[:int(0.6 * sampleCount), :]

        self.DataTrainX = self.DataX[:int(0.8 * sampleCount), :]
        self.DataTrainY = self.DataY[:int(0.8 * sampleCount), :]

        self.DataValX = self.DataX[int(0.6 * sampleCount):int(0.8 * sampleCount), :]
        self.DataValY = self.DataY[int(0.6 * sampleCount):int(0.8 * sampleCount), :]

        self.DataTestX = self.DataX[int(0.8 * sampleCount):, :]
        self.DataTestY = self.DataY[int(0.8 * sampleCount):, :]

        fp.close()

        # print(self.DataTrainX.shape)
        # print(self.DataTrainY.shape)
        #
        # print(self.DataValX.shape)
        # print(self.DataValY.shape)
        #
        # print(self.DataTestX.shape)
        # print(self.DataTestY.shape)

        self.lossSimulator = dataLossSimulator.dataLossSimulator(self.DataX.shape[1], self.lossRate, self.setLossValue)

        self.DataTrainXLoss, self.trainXLossLocations = self.lossSimulator.lossSimulate(self.DataTrainX)
        self.DataValXLoss, self.valXLossLocations = self.lossSimulator.lossSimulate(self.DataValX)
        self.DataTestXLoss, self.testLossLocations = self.lossSimulator.lossSimulate(self.DataTestX)

        self.DataTrainX = self.DataTrainXLoss
        self.DataValX = self.DataValXLoss
        self.DataTestX = self.DataTestXLoss

    def minmax_scale(self):  ###min max scale
        feature =  self.DataTrainX.shape[1]
        sample = self.DataTrainX.shape[0]
        maxarray = self.DataTrainX[0, :].copy()
        minarray = self.DataTrainX[0, :].copy()
        # print maxlist,minlist
        # print maxlist is minlist
        for i in range(sample):
            for j in range(feature):
                if self.DataTrainX[i, j]!=self.setLossValue:
                    if self.DataTrainX[i,j]>maxarray[j]:
                        maxarray[j] = self.DataTrainX[i, j]
                    elif self.DataTrainX[i,j]<minarray[j]:
                        minarray[j] = self.DataTrainX[i, j]

        # print maxlist, minlist
        # self.DataTrainX = (self.DataTrainX - minarray)/(maxarray - minarray)
        # # print self.DataTrainX
        # self.DataTestX = (self.DataTestX - minarray)/(maxarray - minarray)
        # self.DataValX = (self.DataValX - minarray)/(maxarray - minarray)

        self.__scaleprocess(self.DataTrainX, minarray, maxarray)
        self.__scaleprocess(self.DataValX, minarray, maxarray)
        self.__scaleprocess(self.DataTestX, minarray, maxarray)

        # print self.DataTrainX

        return

    def __scaleprocess(self, data, minarray, maxarray):
        for sample in range(data.shape[0]):
            for feature in range(data.shape[1]):
                if data[sample, feature] != self.setLossValue:
                    data[sample, feature] = (data[sample, feature] - minarray[feature])/maxarray[feature]


    def MeanPreProcess(self):
        featureMeanFor = dict()
        validNumFor = dict()
        ylabel = self.DataTrainY.argmax(1)

        for i in range(self.yClassNum):
            featureMeanFor[i] = [0.] * self.DataTrainX.shape[1]
            validNumFor[i] = [0] * self.DataTrainX.shape[1]

        for feature in range(self.DataTrainX.shape[1]):
            for sample in range(self.DataTrainX.shape[0]):
                if self.DataTrainX[sample, feature] != self.setLossValue:
                    featureMeanFor[ylabel[sample]][feature] += self.DataTrainX[sample, feature]
                    validNumFor[ylabel[sample]][feature] += 1

        for yClass in range(self.yClassNum):
            for feature in range(self.DataTrainX.shape[1]):
                featureMeanFor[yClass][feature] /= validNumFor[yClass][feature]

        # print(featureMeanFor)
        featureMean = dict()
        for feature in range(self.DataTrainX.shape[1]):
            summary = 0.
            num = 0
            for yClass in range(self.yClassNum):
                summary += featureMeanFor[yClass][feature] * validNumFor[yClass][feature]
                num += validNumFor[yClass][feature]

            featureMean[feature] = summary / num


        ########### preprocess training data set
        for sample in range(self.DataTrainX.shape[0]):
            for feature in range(self.DataTrainX.shape[1]):
                if self.DataTrainX[sample, feature] == self.setLossValue:
                    # self.DataTrainX[sample, feature] = featureMeanFor[ylabel[sample]][feature]
                    self.DataTrainX[sample, feature] = featureMean[feature]

        ###########preprocess validation data set
        ylabel = self.DataValY.argmax(1)
        for sample in range(self.DataValX.shape[0]):
            for feature in range(self.DataValX.shape[1]):
                if self.DataValX[sample, feature] == self.setLossValue:
                    # self.DataValX[sample, feature] = featureMeanFor[ylabel[sample]][feature]
                    self.DataValX[sample, feature] = featureMean[feature]

        #########preprocess test data set
        ylabel = self.DataTestY.argmax(1)
        for sample in range(self.DataTestX.shape[0]):
            for feature in range(self.DataTestX.shape[1]):
                if self.DataTestX[sample, feature] == self.setLossValue:
                    # self.DataTestX[sample, feature] = featureMeanFor[ylabel[sample]][feature]
                    self.DataTestX[sample, feature] = featureMean[feature]



if __name__ == '__main__':
    # loadiris = loadIris()
    myloadData = loadData('iris.txt', 0.3, -1.)
    # myloadData.MeanPreProcess()
    # print myloadData.DataTrainX, '\n', myloadData.DataTrainY
    myloadData.minmax_scale()