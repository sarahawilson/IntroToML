# Sarah Wilson 
# 303 - 921 - 7225
# Project 3
# Introduction to Machine Learning

import pandas as pd
import math
import numpy as np


class CARTHelper:
    def __init__(self, dataSetName: str,  
                 classHeaderName: str,
                 uniqueToDropHeader,
                 cartAllDataSets
                 ):
        self.name = "CART Helper"
        self.classHeaderName = classHeaderName
        self.dataSetName = dataSetName
        self.dropHeaderName = uniqueToDropHeader
        self.CARTDecTreeRoot = Node()
        self.CARTAllDataSets = cartAllDataSets
        self.epsilon = 10
        
    def dropUniqueIDs(self, dataFrame):
        if(self.dropHeaderName != None):
            resultDF = dataFrame.drop([self.dropHeaderName], axis=1)
        return resultDF
    
    def clearTree(self):
        self.CARTDecTreeRoot = Node()
        
        
    def runCARTAlgo(self, testDF, trainDF):
        print('Running CART - Univariate')
        print('Building Tree')
        self.generateTree(trainDF, self.CARTDecTreeRoot)
        print('End of CART Tree Building')
        
        sumDelta = 0    
        for rowIdx in range(len(testDF)):
             delta = self.runTestDFThroughTree(testDF, self.CARTDecTreeRoot, rowIdx)
             sumDelta = sumDelta + delta
        print('Regression Error:')
        regError = sumDelta / (len(testDF))
        print(regError)
        return self.CARTDecTreeRoot

    def runTestDFThroughTree(self, testDF, currentNode, rowIdx):
        if(len(currentNode.childrenNodes) == 0):
            #Reached a Leaf in the Tree
            #print('Reached a Leaf')
            #print('')
            obsClassValue = testDF[self.classHeaderName].values[rowIdx]
            nodeClassValue = float(currentNode.getNodeContent())
            delta = (obsClassValue - nodeClassValue)**2
            return delta
        
        #Get the feature name stored in the node
        nodeFeature = currentNode.getNodeContent()
        
        #Get the value of the Observation in feature from current row in observations
        obsVal = testDF[nodeFeature].values[rowIdx]
        
        for key in currentNode.childNodePathDict:
            curString = key
            stripString = curString.lstrip('<=')
            stripString = stripString.lstrip('>')
            value = float(stripString)
            
        if(obsVal <= value):
            countStop = 0
        elif(obsVal > value):
            countStop = 1
          
        count = 0
        for key in currentNode.childNodePathDict:
            if (countStop == count):
                useKey = key
            elif (countStop == count):
                useKey = key
            count = count + 1
        
        
        #Get the Child Index in Tree based on Observation Value
        childIdx = currentNode.childNodePathDict[useKey]
        nextNode = currentNode.getChildNode(childIdx)
        classificationValue = self.runTestDFThroughTree(testDF, nextNode, rowIdx)
        return classificationValue




        
    def generateTree(self, currentPartition, currentNode):
        meanPartition = currentPartition[self.classHeaderName].mean()
        MSEPartition = self._calcMSE(meanPartition, currentPartition)
        if (MSEPartition <= self.epsilon):
            currentNode.setNodeContent(str(meanPartition))
            return
        
        bestFeatureList = self._determineBestFeatureForSplit(currentPartition)
        if (bestFeatureList[0] == None):
            #Hit a case where there were no possible splits
            #Just create a node
            currentNode.setNodeContent(str(meanPartition))
            return
            
        
        
        bestFeatureName = bestFeatureList[0]
        bestFeatureMean = bestFeatureList[1]
        currentNode.setNodeContent(bestFeatureName)

        numChildren = 2        
        childIdx = 0
        for numIdx in range(numChildren):
            stringVal = str(bestFeatureMean)
            if(childIdx == 0):
                pathName = "<=" + stringVal
            elif(childIdx == 1):
                pathName = ">" + stringVal
            else:
                print('WARNING!!!!!!')
                print('More than two nodes for feature type Numeric')
                print('WARNING!!!!!!')
            currentNode.addChildNode()
            currentNode.childNodePathDict[pathName] = childIdx
                
            #Want to build a new parition of that data that 
            #only includes instances where that Feature has the Attribue less  than or equal to
            #the split value 
            if(childIdx == 0):
                newPartition = currentPartition[(currentPartition[bestFeatureName] <= bestFeatureMean)]
            elif(childIdx == 1):
                newPartition = currentPartition[(currentPartition[bestFeatureName] > bestFeatureMean)]
            if(len(newPartition) == 0):
                print('WARNING!!!!!!')
                print('WHOA WHOA WHOA')
                print('something is not right')
            newBaseNode = currentNode.getChildNode(childIdx)
            childIdx = childIdx + 1;
            #print('Debug Break point prior to entering recursive call')
            self.generateTree(newPartition, newBaseNode)
    
    def _dropFeaturesWithOutLeftRight(self, currentPartition):
        listToDrop = []
        for featureName in currentPartition:
            meanOfFeature = currentPartition[featureName].mean()
            leftDF_feature = currentPartition[(currentPartition[featureName] <= meanOfFeature)]
            rightDF_feature = currentPartition[(currentPartition[featureName] > meanOfFeature)]
            if((len(leftDF_feature) == 0) or(len(rightDF_feature) == 0)):
                #Ignore this feature as calculation for best 
                listToDrop.append(featureName)
        
        resultDF = currentPartition.drop(listToDrop, axis=1)
        return resultDF    
        
        
    def _determineBestFeatureForSplit(self, currentPartition):
        featureSplitError = {}
        bestFeatureSplitValDict = {}
        bestFeatureMeanList = []
        for featureName in currentPartition:
            if (featureName == self.classHeaderName):
                #Skip Class Header Name as candidate
                continue
            
            meanOfFeature = currentPartition[featureName].mean()
                
            leftDF_feature = currentPartition[(currentPartition[featureName] <= meanOfFeature)]
            rightDF_feature = currentPartition[(currentPartition[featureName] > meanOfFeature)]
            if((len(leftDF_feature) == 0) or(len(rightDF_feature) == 0)):
                #Ignore this feature as calculation for best 
                continue
                
            #Get the mean of the class
            meanLeft_class = leftDF_feature[self.classHeaderName].mean()
            meanRight_class = rightDF_feature[self.classHeaderName].mean()
            
            sumError_Left = self._calcSumSquareError(meanLeft_class, leftDF_feature)
            sumError_Right = self._calcSumSquareError(meanRight_class, rightDF_feature)
            
            sumLeftRight = sumError_Left + sumError_Right
            splitError = sumLeftRight / len(currentPartition)
            
            featureSplitError[featureName] = splitError
            bestFeatureSplitValDict[featureName] = meanOfFeature
  
        if(len(bestFeatureSplitValDict) == 0):
            bestFeatureMeanList.append(None)
            bestFeatureMeanList.append(None)
            return bestFeatureMeanList
        bestFeatureName = max(featureSplitError, key=featureSplitError.get)
        bestFeatureMeanList.append(bestFeatureName)
        bestFeatureMeanList.append( bestFeatureSplitValDict[bestFeatureName] )
        return bestFeatureMeanList

    def _calcMSE(self, mean, runOnDataFrame):
        sumDelta = 0
        numDataPoints = len(runOnDataFrame)
        if (numDataPoints == 0):
            print('Stop')
        for rowIdx in range(len(runOnDataFrame)):
            curClassVal = runOnDataFrame[self.classHeaderName].values[rowIdx]
            delta = (curClassVal - mean)**2
            sumDelta = sumDelta + delta
        MSE = sumDelta / numDataPoints
        return MSE
    
    def _calcSumSquareError(self, mean, runOnDataFrame):
        sumDelta = 0
        for rowIdx in range(len(runOnDataFrame)):
            curClassVal = runOnDataFrame[self.classHeaderName].values[rowIdx]
            delta = (curClassVal - mean)**2
            sumDelta = sumDelta + delta
        return sumDelta
        
        
    
class Node:
    def __init__(self):
        self.nodeContent = None
        self.childrenNodes = []
        self.childNodePathDict = {}  #Key will be option name, value will be childIdx
        
    def setNodeContent(self, nodeLabel: str):
        self.nodeContent = nodeLabel;
        
    def getNodeContent(self):
        return self.nodeContent
    
    def addChildNode(self):
        newChild = Node()
        self.childrenNodes.append(newChild)
    
    def getChildNode(self, inputNodeChildIdx):
        return self.childrenNodes[inputNodeChildIdx]