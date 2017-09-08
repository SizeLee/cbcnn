#coding:utf-8
import numpy as np
import myLoadData
import combineNumCalculate

class combineFeature:
    # __featureNum = 0
    # __combineNum = 0
    # __featureCombineMap = []

    def __init__(self, featureNum, combineNum):
        self.__featureNum = 0
        self.__combineNum = 0
        self.__featureCombineMap = []

        self.__featureNum = featureNum
        self.__combineNum = combineNum

        if self.__featureNum < self.__combineNum:
            print("Warning in combineFeature: feature Num is smaller combine Num, No combine will be make\n")

        self.__combineFun(0, 0, )

    def __combineFun(self, considerFeatureNo, alreadyCombine, *combine):
        # print(*combine)
        if alreadyCombine == self.__combineNum:
            # print(combine)
            self.__featureCombineMap.append(combine)
            return
        elif considerFeatureNo >= self.__featureNum:
            return

        self.__combineFun(considerFeatureNo+1, alreadyCombine, *combine)

        # self.__combineFun(considerFeatureNo + 1, alreadyCombine + 1, *combine, considerFeatureNo)
        self.__combineFun(considerFeatureNo + 1, alreadyCombine + 1, *combine + (considerFeatureNo,))

    def outputCombineMap(self):
        # self.__combineFun(0,0,)
        print(self.__featureCombineMap)

    def getFeatureNum(self):
        return self.__featureNum

    def getCombineNum(self):
        return self.__combineNum

    def getCombKindNum(self):
        return combineNumCalculate.combineNumCal(self.__featureNum, self.__combineNum)

    def makeCombineData(self, dataX):
        # print(dataX)
        # print(dataX.shape[0])

        combineData = [] ##所有样本的所有组合
        for sample in dataX:
            combineSample = [] ##一个样本的所有组合
            # print(sample)
            # print(len(sample.shape))
            if len(sample.shape) > 2:
                print('Error in combineFeature: Can\'t support combine Data dim large than 3')

            elif len(sample.shape) == 2:
                if sample.shape[0] > 1 and sample.shape[1] > 1:
                    print('Error in  combineFeature: Can\'t support combine Data Sample isn\'t line vector')

                elif sample.shape[0] > 1 and sample.shape[1] == 1:
                    sample = sample.transpose()[0]
                    # print(sample)


            for combine in self.__featureCombineMap:
                # print(combine)
                combineTemp = [] ###一个组合
                for column in combine:
                    # print(column)
                    combineTemp.append(sample[column])

                combineSample.append(combineTemp)

            # print(np.array(combineSample))
            # break;
            combineData.append(combineSample)
            # break;
        # print(combineData)

        return np.array(combineData)

    #todo def bp delt map generator
    def BP(self, sensitivityFactor):
        if sensitivityFactor.shape[1] != len(self.__featureCombineMap) \
                or sensitivityFactor.shape[2] != len(self.__featureCombineMap[0]):
            print('Error in combineFeature BP: sensitivity factor size error\n')
            exit(1)  # todo throw out error

        sampleNum = sensitivityFactor.shape[0]
        combNum = sensitivityFactor.shape[1]
        combLengthFeatureNum = sensitivityFactor.shape[2]

        formerSF = np.zeros((sampleNum, self.__featureNum, 1))

        for i in range(sampleNum):
            for j in range(combNum):
                for k in range(combLengthFeatureNum):
                    formerSF[i, self.__featureCombineMap[j][k]] += sensitivityFactor[i, j, k]

        # print(sensitivityFactor)
        # print(self.__featureCombineMap)
        # print(formerSF)
        return formerSF


class NewCombineFeature:
    # __featureNum = 0
    # __combineNum = 0
    # __featureCombineMap = []

    def __init__(self, featureNum, combineNum, setLossLabel):
        self.__featureNum = 0
        self.__combineNum = 0
        self.__featureCombineMap = []
        self.__featureConnection = []
        self.__poolingPosition = None
        self.__outputDataX = None
        self.__lossLabel = setLossLabel

        for i in range(featureNum):
            self.__featureConnection.append([])

        self.__featureNum = featureNum
        self.__combineNum = combineNum

        if self.__featureNum < self.__combineNum:
            print("Warning in combineFeature: feature Num is smaller combine Num, No combine will be make\n")

        self.__combCount = combineNumCalculate.combineNumCal(self.__featureNum, self.__combineNum)

        self.__combineFun(0, 0, )

    def __combineFun(self, considerFeatureNo, alreadyCombine, *combine):
        # print(*combine)
        if alreadyCombine == self.__combineNum:
            # print(combine)
            self.__featureCombineMap.append(combine)
            for i in combine:
                self.__featureConnection[i].append((len(self.__featureCombineMap) - 1))

            return
        elif considerFeatureNo >= self.__featureNum:
            return

        self.__combineFun(considerFeatureNo + 1, alreadyCombine, *combine)

        # self.__combineFun(considerFeatureNo + 1, alreadyCombine + 1, *combine, considerFeatureNo)
        self.__combineFun(considerFeatureNo + 1, alreadyCombine + 1, *combine + (considerFeatureNo,))

    def outputCombineMap(self):
        # self.__combineFun(0,0,)
        print(self.__featureCombineMap)

    def getFeatureNum(self):
        return self.__featureNum

    def getCombineNum(self):
        return self.__combineNum

    def getCombKindNum(self):
        return combineNumCalculate.combineNumCal(self.__featureNum, self.__combineNum)

    def makeCombineData(self, dataX, lossLocations):
        # print(dataX)
        # print(dataX.shape[0])

        combineData = [] ##所有样本的所有组合
        for sample in dataX:
            combineSample = [] ##一个样本的所有组合
            # print(sample)
            # print(len(sample.shape))
            if len(sample.shape) > 2:
                print('Error in combineFeature: Can\'t support combine Data dim large than 3')

            elif len(sample.shape) == 2:
                if sample.shape[0] > 1 and sample.shape[1] > 1:
                    print('Error in  combineFeature: Can\'t support combine Data Sample isn\'t line vector')

                elif sample.shape[0] > 1 and sample.shape[1] == 1:
                    sample = sample.transpose()[0]
                    # print(sample)


            for combine in self.__featureCombineMap:
                # print(combine)
                combineTemp = [] ###一个组合
                zerosFlag = False
                for column in combine:
                    # print(column)
                    if sample[column] == self.__lossLabel:
                        zerosFlag = True
                        break

                    combineTemp.append(sample[column])

                if zerosFlag is True:
                    combineTemp = np.zeros(len(combineTemp))

                combineSample.append(combineTemp)

            # print(np.array(combineSample))
            # break;
            combineData.append(combineSample)
            # break;
        # print(combineData)

        return np.array(combineData)

    #todo def bp delt map generator
    def BP(self, sensitivityFactor):
        if sensitivityFactor.shape[1] != len(self.__featureCombineMap) \
                or sensitivityFactor.shape[2] != len(self.__featureCombineMap[0]):
            print('Error in combineFeature BP: sensitivity factor size error\n')
            exit(1)  # todo throw out error

        sampleNum = sensitivityFactor.shape[0]
        combNum = sensitivityFactor.shape[1]
        combLengthFeatureNum = sensitivityFactor.shape[2]

        formerSF = np.zeros((sampleNum, self.__featureNum, 1))

        for i in range(sampleNum):
            for j in range(combNum):
                for k in range(combLengthFeatureNum):
                    formerSF[i, self.__featureCombineMap[j][k]] += sensitivityFactor[i, j, k]

        # print(sensitivityFactor)
        # print(self.__featureCombineMap)
        # print(formerSF)
        return formerSF


class maxCombineFeature:
    def __init__(self, featureNum, combineNum):
        self.__featureNum = 0
        self.__combineNum = 0
        self.__featureCombineMap = []
        self.__featureConnection = []
        self.__poolingPosition = None
        self.__outputDataX = None

        for i in range(featureNum):
            self.__featureConnection.append([])

        self.__featureNum = featureNum
        self.__combineNum = combineNum

        if self.__featureNum < self.__combineNum:
            print("Warning in combineFeature: feature Num is smaller combine Num, No combine will be make\n")

        self.__combCount = combineNumCalculate.combineNumCal(self.__featureNum, self.__combineNum)

        self.__combineFun(0, 0, )

    def __combineFun(self, considerFeatureNo, alreadyCombine, *combine):
        # print(*combine)
        if alreadyCombine == self.__combineNum:
            # print(combine)
            self.__featureCombineMap.append(combine)
            for i in combine:
                self.__featureConnection[i].append((len(self.__featureCombineMap) - 1))

            return
        elif considerFeatureNo >= self.__featureNum:
            return

        self.__combineFun(considerFeatureNo+1, alreadyCombine, *combine)

        # self.__combineFun(considerFeatureNo + 1, alreadyCombine + 1, *combine, considerFeatureNo)
        self.__combineFun(considerFeatureNo + 1, alreadyCombine + 1, *combine + (considerFeatureNo,))



    def combineAndPooling(self, data):
        dataArgSort = np.argsort(data, axis=1)
        self.__outputDataX = np.ones((data.shape[0], self.__combCount)) * (-1)
        self.__poolingPosition = np.zeros((data.shape[0], self.__combCount), dtype=int)
        for sample in range(dataArgSort.shape[0]):
            for rankindex in range(self.__featureNum - self.__combineNum + 1):

                opfeatureindex = dataArgSort[sample, -1-rankindex, 0]

                for combindex in self.__featureConnection[opfeatureindex]:
                    if self.__outputDataX[sample, combindex] == -1:
                        self.__outputDataX[sample, combindex] = data[sample, opfeatureindex, 0]
                        self.__poolingPosition[sample, combindex] = int(opfeatureindex)

        return self.__outputDataX

    def BP(self, sensitivityFactor):

        sampleNum = sensitivityFactor.shape[0]
        combNum = sensitivityFactor.shape[1]

        formerSF = np.zeros((sampleNum, self.__featureNum, 1))

        for i in range(sampleNum):
            for j in range(combNum):
                formerSF[i, self.__poolingPosition[i, j], 0] += sensitivityFactor[i, j]

        # print(sensitivityFactor)
        # print(self.__featureCombineMap)
        # print(formerSF)
        return formerSF

    def outputCombineMap(self):
        # self.__combineFun(0,0,)
        print(self.__featureCombineMap)
        print self.__featureConnection



if __name__ == '__main__':
    # a = combineFeature(4,2)
    # a.outputCombineMap()
    #
    # a = combineFeature(4, 2)
    # a.outputCombineMap()
    # irisData = myLoadData.loadIris()
    # b = a.makeCombineData(irisData.DataTrainX)
    # print(b)
    # print(b.shape)
    a = maxCombineFeature(6, 4)
    a.outputCombineMap()




