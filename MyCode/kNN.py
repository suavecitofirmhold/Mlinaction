from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group, labels
    
def classfy0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  #行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #axis=0按行axis=1是按列执行
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2Matrix(fileName):
    fr = open(fileName)
    arrayoLines = fr.readlines()
    numberOfLine = len(arrayoLines)
    returnMat = zeros((numberOfLine, 3))
    classLabelVector = []
    index = 0
    for line in arrayoLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def img2vector(filename):
    retVec = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            retVec[0,32*i+j]=int(lineStr[j])
    return retVec

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('G:\\machinelearning\\机器学习实战\\《机器学习实战》Python3代码\\machinelearninginaction3x-master\\Ch02\\digits\\trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i:] = img2vector('G:\machinelearning\\机器学习实战\\《机器学习实战》Python3代码\\machinelearninginaction3x-master\\Ch02\\digits\\trainingDigits/%s' % fileNameStr)
    testFileList = listdir('G:\\machinelearning\\机器学习实战\\《机器学习实战》Python3代码\\machinelearninginaction3x-master\\Ch02\\digits\\testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('G:\\machinelearning\\机器学习实战\\《机器学习实战》Python3代码\\machinelearninginaction3x-master\\Ch02\\digits\\testDigits\%s' % fileNameStr)
        classifierResult = classfy0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d , the real answer is: %d"\
            % (classifierResult, classNumStr))
        if (classifierResult != classNumStr) : errorCount += 1
    print ("\nthe total number of errors is %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

def main():
    yes = img2vector('G:\\machinelearning\\机器学习实战\\《机器学习实战》Python3代码\\machinelearninginaction3x-master\\Ch02\\digits\\testDigits\\0_0.txt')
    #datingDataMat,datingLabels = file2Matrix('G:\machinelearning\机器学习实战\《机器学习实战》Python3代码\machinelearninginaction3x-master\Ch02\datingTestSet.txt')
    #group,labels = createDataSet()
    #classfy0([0,0],group,labels,3)

if __name__ == '__main__':
    #yes = img2vector('G:\\machinelearning\\机器学习实战\\《机器学习实战》Python3代码\\machinelearninginaction3x-master\\Ch02\\digits\\testDigits\\0_0.txt')
    # print(__name__)
    handwritingClassTest()
