from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from MyCombCNNPack import accuracyEvaluate

class myJudge:
    def __init__(self, yClassDic, yPredict, yReal):
        if yPredict is None or yReal is None:
            raise AttributeError

        if not isinstance(yPredict, np.ndarray) and not isinstance(yPredict, list):
            raise AttributeError

        if not isinstance(yReal, np.ndarray) and not isinstance(yReal, list):
            raise AttributeError

        if isinstance(yPredict, np.ndarray):
            if len(yPredict.shape) > 2:
                raise AttributeError

            if len(yPredict.shape) == 2:
                if yPredict.shape[1] == 1:
                    yPredict = yPredict.T

            yPredict = yPredict.tolist()

        if isinstance(yReal, np.ndarray):
            if len(yReal.shape) > 2:
                raise AttributeError

            if len(yReal.shape) == 2:
                if yReal.shape[1] == 1:
                    yReal = yReal.T

            yReal = yReal.tolist()

        self.yPredict = yPredict
        self.yReal = yReal

        self.labelDic = {v: k for k, v in yClassDic.items()}

        self.classNum = len(self.labelDic)

        self.myconfusionmatrix = np.zeros((self.classNum, self.classNum))
        for real, pre in zip(self.yReal, self.yPredict):
            self.myconfusionmatrix[real, pre] += 1

        self.labels = []
        for i in range(self.classNum):
            self.labels.append(self.labelDic[i])

    def plotConfuseMatrix(self):
        plt.figure()
        plt.imshow(self.myconfusionmatrix, cmap=plt.cm.binary, interpolation='nearest')

        width = self.classNum
        height = self.classNum

        cb = plt.colorbar()
        tickRange = np.array(range(width))
        plt.xticks(tickRange, self.labels)
        plt.yticks(tickRange, self.labels, rotation=90)
        plt.xlabel('Predicted label')
        plt.ylabel('Real label')

        ind_array = np.arange(self.classNum)
        x, y = np.meshgrid(ind_array, ind_array)
        # print(x,y)
        for pre, real in zip(x.flatten(), y.flatten()):
            num = self.myconfusionmatrix[real, pre]
            if num > 0.000001:
                plt.text(pre, real, "%0.2d" % (num,), color='blue', fontsize=16, va='center', ha='center')

        plt.savefig('confusionMatrix.png')
        plt.close()


    def getRecall(self):
        recallDic = {v: k for k, v in self.labelDic.items()}
        for eachKey in recallDic:
            recallDic[eachKey] = 0.

        sumRealMatrix = self.myconfusionmatrix.sum(axis= 1)

        for i in range(self.classNum):
            if sumRealMatrix[i] == 0:
                recallRate = 0.
                
            else:
                recallRate = self.myconfusionmatrix[i, i]/float(sumRealMatrix[i])

            recallDic[self.labelDic[i]] = recallRate

        return recallDic


    def getPrecision(self):
        precisionDic = {v: k for k, v in self.labelDic.items()}
        for eachKey in precisionDic:
            precisionDic[eachKey] = 0.

        sumPredictMatrix = self.myconfusionmatrix.sum(axis= 0)

        for i in range(self.classNum):
            if sumPredictMatrix[i] == 0:
                precisionRate = 0.

            else:
                precisionRate = self.myconfusionmatrix[i, i]/float(sumPredictMatrix[i])

            precisionDic[self.labelDic[i]] = precisionRate

        return precisionDic


    def calculateF1(self):
        f1Dic = {v: k for k, v in self.labelDic.items()}
        for eachKey in f1Dic:
            f1Dic[eachKey] = 0.

        precisionD = self.getPrecision()
        recallD = self.getRecall()

        for eachKey in f1Dic:
            if precisionD[eachKey] + recallD[eachKey] == 0.:
                f1Dic[eachKey] = 0.
            else:
                f1Dic[eachKey] = 2 * precisionD[eachKey] * recallD[eachKey] / (precisionD[eachKey] + recallD[eachKey])

        return f1Dic


    def getTPTNFPFN(self): ##get true positive, true negative, false positive, false negative
        TTFFDic = {v: k for k, v in self.labelDic.items()}
        for eachKey in TTFFDic:
            TTFFDic[eachKey] = dict()

        sumRealMatrix = self.myconfusionmatrix.sum(axis=1)
        sumPredictMatrix = self.myconfusionmatrix.sum(axis=0)

        for i in range(self.classNum):
            TTFFDic[self.labelDic[i]]['TruePositive'] = self.myconfusionmatrix[i, i]
            TTFFDic[self.labelDic[i]]['FalsePositive'] = sumPredictMatrix[i] - self.myconfusionmatrix[i, i]
            TTFFDic[self.labelDic[i]]['FalseNegative'] = sumRealMatrix[i] - self.myconfusionmatrix[i, i]
            sampleNum = sumRealMatrix.sum()
            TTFFDic[self.labelDic[i]]['TrueNegative'] = sampleNum - TTFFDic[self.labelDic[i]]['TruePositive'] - \
                                                                    TTFFDic[self.labelDic[i]]['FalsePositive'] - \
                                                                    TTFFDic[self.labelDic[i]]['FalseNegative']

        return TTFFDic


    def getAccuracy(self):
        ac = accuracyEvaluate.AccuracyRate(self.yPredict, self.yReal)

        return ac



