# Sarah Wilson 
# 303 - 921 - 7225
# Project 3
# Introduction to Machine Learning

import pandas as pd
import math
import numpy as np


class ID3Helper_ROUND2:
    def __init__(self, dataSetName: str, 
                 numClassProblem: int, 
                 classHeaderName: str,
                 uniqueToDropHeader,
                 id3AllDataSets
                 ):
        self.name = "ID3 Helper"
        self.treeRootNode = Node()
        self.numClassProblem = numClassProblem
        self.classHeaderName = classHeaderName
        self.dataSetName = dataSetName
        self.dropHeaderName = uniqueToDropHeader
        self.ID3DecTreeRoot = Node()
        self.ID3AllDataSets = id3AllDataSets

    def runID3Algo(self, testDF, trainDF):
        print('Running ID3 - Univariate')
        print('Building Tree')
        self.generateTree(trainDF, self.ID3DecTreeRoot)
        print('End of ID3 Tree Building')
        
        #TODO: Insert logic for how to handle dropping item to classify through tree
        #self.runTestDFThroughTree(testDF)
        #return self.ID3DecTreeRoot

    def generateTree(self, currentPartition, currentNode):
        #Check if the currentPartition only has one Class Label in it
        #if so return a leaf
        classCount = currentPartition[self.classHeaderName].value_counts()
        if(len(classCount) == 1):
            currentNode.setNodeContent(classCount.index[0])
            return
        
        maxGRFeatureName = self._determineMaxGainRatioFeature(currentPartition)
        currentNode.setNodeContent(maxGRFeatureName)
        featureType = self._getFeatureType(maxGRFeatureName)
        
        #Numeric Split 
        if (featureType == 'Num'):
            #TODO:
            # Need to determine the best split value c 
            spltValList = self._determineSplitValueList(currentPartition, maxGRFeatureName)
            bestSplitValue = self._determineMaxGainRatioSplit(currentPartition, spltValList)
            
            numChildren = 2
            childIdx = 0
            for numIdx in range(numChildren):
                stringVal = str(bestSplitValue)
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
                    newPartition = currentPartition[(currentPartition[maxGRFeatureName] <= bestSplitValue)]
                elif(childIdx == 1):
                    newPartition = currentPartition[(currentPartition[maxGRFeatureName] > bestSplitValue)]
                if(len(newPartition) == 0):
                    print('WARNING!!!!!!')
                    print('WHOA WHOA WHOA')
                    print('something is not right')
                newBaseNode = currentNode.getChildNode(childIdx)
                childIdx = childIdx + 1;
                print('Debug Break point prior to entering recursive call')
                self.generateTree(newPartition, newBaseNode)
                
                    
        #Categorical Split
        elif(featureType == 'Cat'):
            #Determine the Range of the Feature 
            #Pull it's unique attributes
            featureOptions = currentPartition[maxGRFeatureName].unique()
            childIdx = 0;
            for option in featureOptions:
                currentNode.addChildNode()
                currentNode.childNodePathDict[option] = childIdx;
                
                #Want to build a new parition of that data that 
                #only includes instances where that Feature has the Attribue
                newPartition = currentPartition[(currentPartition[maxGRFeatureName] == option)]
                newBaseNode = currentNode.getChildNode(childIdx)
                childIdx = childIdx + 1;
                print('Debug Break point prior to entering recursive call')
                self.generateTree(newPartition, newBaseNode)

    def _determineMaxGainRatioFeature(self, currentPartition): 
        entPar = self._calcEntropy(currentPartition)
        expEnt = self._calcExpEntropyAllFeaturesInCurrentParition(currentPartition)
        gainPar = self._calcGainAllFeaturesInCurrentParition(entPar, expEnt)
        infoValPar = self._calcInformationValueAllFeaturesInCurretPartition(currentPartition)
        gainRatio = self._calGainRatioAllFeaturesInCurrentPartition(gainPar, infoValPar)
        
        maxGainRatioFeature = max(gainRatio, key=gainRatio.get)
        print('Feature with Max Gain Ratio: \t' + maxGainRatioFeature)
        return maxGainRatioFeature
   

    def _calcGainAllFeaturesInCurrentParition(self, entropyOfPartition, expectedEntropyAllFeatures: dict):
        #TODO: Add comments
        gainAllFeatures = {}
        for featureName in expectedEntropyAllFeatures:
            featureExpectedEntropy = expectedEntropyAllFeatures[featureName]
            gainAllFeatures[featureName] = entropyOfPartition - featureExpectedEntropy
        #print('GAIN!!')
        #print(gainAllFeatures)
        return gainAllFeatures
    
    def _calGainRatioAllFeaturesInCurrentPartition(self, gainAllFeatures, ivAllFeatures):
        gainRatioAllFeatures = {}
        for featureName in gainAllFeatures:
            curFeatureGain = gainAllFeatures[featureName]
            curFeatureIV = ivAllFeatures[featureName]
            if(curFeatureIV == 0):
                print('WARNING encountered divide by zero')
                print(featureName)
                curGainRatio = 0
            else:
                curGainRatio = curFeatureGain / curFeatureIV
            gainRatioAllFeatures[featureName] = curGainRatio
        
        #print('GAIN RATIO!!')
        #print(gainRatioAllFeatures)
        return(gainRatioAllFeatures)
    
    
    def _calcEntropy(self, runOnDataFrame):
        entropyI = None
        if(self.numClassProblem == 2):
            classOptionCounts = runOnDataFrame[self.classHeaderName].value_counts()

            #Class 1 Option Name and Count
            opt1Name = classOptionCounts.index[0]
            opt1Count = classOptionCounts[opt1Name]

            #Class 2 Option Name and Count
            if(classOptionCounts.size == 1):
                opt2Count = 0
            else:
                opt2Name = classOptionCounts.index[1]
                opt2Count = classOptionCounts[opt2Name]
            
            termOpt1 = ((-opt1Count/(opt1Count + opt2Count))* math.log2(opt1Count/(opt1Count + opt2Count)))
            
            if(classOptionCounts.size == 1):
                termOpt2 = 0
            else:
                termOpt2 = ((opt2Count/(opt1Count + opt2Count))* math.log2(opt2Count/(opt1Count + opt2Count)))
            
            entropyI = termOpt1 - termOpt2
        return entropyI


    def _determineSplitValueList(self, currentPartition, useFeatureName):
        sortedPartition = currentPartition.sort_values(by=[useFeatureName])
        sortedPartition['Count Class Change'] = sortedPartition[self.classHeaderName].ne(sortedPartition[self.classHeaderName].shift()).cumsum()
        sortedPartition['Change Occured'] = sortedPartition['Count Class Change'].diff()
        sortedPartition['Before Change Occured'] = 0
        sortedPartition['Before Change Occured'][:-1] = sortedPartition['Change Occured'][1:]
        beforeChangeDF = sortedPartition[sortedPartition['Before Change Occured']==1]
        afterChangeDF = sortedPartition[sortedPartition['Change Occured']==1]
        
        possSplitVals = []
        for rowIdx in range(len(beforeChangeDF)):
            beforeVal = beforeChangeDF[useFeatureName].values[rowIdx]
            afterVal = afterChangeDF[useFeatureName].values[rowIdx]
            midPointVal = (beforeVal + afterVal)/2
            possSplitVals.append(midPointVal)
                  
        uniquePossSplits = np.unique(np.array(possSplitVals))
        uniquePossSplitsList = uniquePossSplits.tolist()
        return uniquePossSplitsList
    
    def _calcExpEntropyAllFeaturesInCurrentParition(self, currentPartition):
        expectedEntropyAllFeatures = {}
        for featureName in currentPartition:
            if (featureName == self.classHeaderName):
                continue
            else:
                featureType = self._getFeatureType(featureName)
                if(featureType == 'Num'):
                    #Determine the Unique Split Values for the Numeric Feature
                    splitValueList = self._determineSplitValueList(currentPartition, featureName)
                    curFeatOptionsProb = self._calcProbabilityOnNumOptions(currentPartition, featureName, splitValueList)
                    curFeatOptionsEntropy = self._calcEntropyOnNumOptions(currentPartition, featureName, splitValueList)
                elif(featureType == 'Cat'):
                    curFeatOptionsProb = self._calcProbabilityOnCatOptions(currentPartition, featureName)
                    curFeatOptionsEntropy = self._calcEntropyOnCatOptions(currentPartition, featureName)
            entropySum = 0 
                for index in range(len(curFeatOptionsProb)):
                    curProb = curFeatOptionsProb[index]
                    curEntrop = curFeatOptionsEntropy[index]
                    multTerm = curProb*curEntrop
                    entropySum = entropySum + multTerm
                expectedEntropyAllFeatures[featureName] = entropySum
        #print(expectedEntropyAllFeatures)
        return expectedEntropyAllFeatures
    
    def _calcInformationValueAllFeaturesInCurretPartition(self, currentPartition):
        informationValueAllFeatures = {}
        for featureName in currentPartition:
            if (featureName == self.classHeaderName):
                continue
            else:
                featureType = self._getFeatureType(featureName)
                if(featureType == 'Num'):
                    #Determine the Unique Split Values for the Numeric Feature
                    splitValueList = self._determineSplitValueList(currentPartition, featureName)
                    curFeatOptionInfoValue = self._calcInfoValueOnNumOptions(currentPartition, featureName, splitValueList)
                elif(featureType == 'Cat'):
                    curFeatOptionInfoValue = self._calcInfoValueOnCatOptions(currentPartition, featureName)
                      
                ivSum = 0
                for index in range(len(curFeatOptionInfoValue)):
                    curIV = curFeatOptionInfoValue[index]
                    ivSum = curIV + ivSum
                ivSum = -1*ivSum
                informationValueAllFeatures[featureName] = ivSum
        #print('IV!!')
        #print(informationValueAllFeatures)
        return informationValueAllFeatures
    
    def _calcInfoValueOnCatOptions(self, currentPartition, featureName):
        infoValueAllOptionsInCurFeature = []
        numberObservations = len(currentPartition.index)
        featureOptions = currentPartition[featureName].unique()
        for option in featureOptions:
            optionDF = currentPartition.loc[currentPartition[featureName] == option]
            numberOfOccurancesOption = len(optionDF.index)
            infoValueOption = ((numberOfOccurancesOption/(numberObservations))* math.log2(numberOfOccurancesOption/(numberObservations)))
            infoValueAllOptionsInCurFeature.append(infoValueOption)
        return infoValueAllOptionsInCurFeature  
    
    def _calcInfoValueOnNumOptions(self, currentPartition, featureName, spltiValueList):
        infoValueAllSplitsInCurFeature = []
        numberObservations = len(currentPartition.index)
        for splitVal in splitValueList:
           greaterThanEqToDF = currentPartition[currentPartition[featureName] >= splitVal] 
           numberOfOccurancesOption = len(optionDF.index)
           infoValueOption = ((numberOfOccurancesOption/(numberObservations))* math.log2(numberOfOccurancesOption/(numberObservations)))
           infoValueAllOptionsInCurFeature.append(infoValueOption)
        return infoValueAllOptionsInCurFeature         
    
    
    def _calcProbabilityOnNumOptions(self, currentPartition, featureName, splitValueList):
        probsAllSplitsInCurFeature = []
        numberObservations = len(currentPartition.index)
        for splitVal in splitValueList:
            greaterThanEqToDF = currentPartition[currentPartition[featureName] >= splitVal]
            optionCount = len(greaterThanEqToDF)
            probibilitySplit = optionCount / numberObservations
            probsAllSplitsInCurFeature.append(probibilitySplit)
        return probsAllSplitsInCurFeature
    
    #Calculate the Entropy on Numeric Options    
    def _calcEntropyOnNumOptions(self, currentPartition, featureName, splitValList): 
        entropyAllSplitsInCurFeature = []
        for splitVal in splitValList:
            greaterThanEqToDF = currentPartition[currentPartition[featureName] >= splitVal]
            entropyAllSplitsInCurFeature.append(self._calcEntropy(greaterThanEqToDF))
        return entropyAllSplitsInCurFeature
    
    #Calcualte the Probability on Categorical Options
    def _calcProbabilityOnCatOptions(self, currentPartition, featureName):
        probsAllOptionsInCurFeature = []
        numberObservations = len(currentPartition.index)
        featureOptions = currentPartition[featureName].unique()
        featureOptionCounts = currentPartition[featureName].value_counts()
        #print(featureOptions)
        for option in featureOptions:
            optionCount = featureOptionCounts[option]
            probibilityOption = optionCount / numberObservations
            probsAllOptionsInCurFeature.append(probibilityOption)   
        return probsAllOptionsInCurFeature
    
    #Calculate the Entropy on Categorical Options
    def _calcEntropyOnCatOptions(self, currentPartition, featureName):
        entropyAllOptionsInCurFeature = []
        featureOptions = currentPartition[featureName].unique()
        for option in featureOptions:
            optionDF = currentPartition.loc[currentPartition[featureName] == option]
            entropyAllOptionsInCurFeature.append(self._calcEntropy(optionDF))
        return entropyAllOptionsInCurFeature
    
    
    
    
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