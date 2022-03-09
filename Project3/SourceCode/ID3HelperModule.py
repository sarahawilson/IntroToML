# Sarah Wilson 
# 303 - 921 - 7225
# Project 3
# Introduction to Machine Learning

import pandas as pd
import math
import numpy as np

class ID3Helper:
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
        
    def dropUniqueIDs(self, dataFrame):
        if(self.dropHeaderName != None):
            resultDF = dataFrame.drop([self.dropHeaderName], axis=1)
        return resultDF
    
    def _calcPartitionEntropy(self, currentPartition):
        entropyI_Pi = None
        if(self.numClassProblem == 2):
            classOptions = currentPartition[self.classHeaderName].unique()
            print(classOptions)
            classOptionCounts = currentPartition[self.classHeaderName].value_counts()
            print(classOptionCounts)
            #Class 1 Option Name and Count
            opt1Name = classOptionCounts.index[0]
            opt1Count = classOptionCounts[opt1Name]
            
            #Class 2 Option Name and Count
            opt2Name = classOptionCounts.index[1]
            opt2Count = classOptionCounts[opt2Name]
            
            termOpt1 = ((-opt1Count/(opt1Count + opt2Count))* math.log2(opt1Count/(opt1Count + opt2Count)))
            termOpt2 = ((opt2Count/(opt1Count + opt2Count))* math.log2(opt2Count/(opt1Count + opt2Count)))
            
            entropyI_Pi = termOpt1 - termOpt2
            print(entropyI_Pi)
            
        return entropyI_Pi 
    
    def _calcGainAllFeaturesInCurrentParition(self, entropyOfPartition, expectedEntropyAllFeatures: dict):
        #TODO: Add comments
        gainAllFeatures = {}
        for featureName in expectedEntropyAllFeatures:
            featureExpectedEntropy = expectedEntropyAllFeatures[featureName]
            gainAllFeatures[featureName] = entropyOfPartition - featureExpectedEntropy
        print('GAIN!!')
        print(gainAllFeatures)
        return gainAllFeatures
    
    def _calcInformationValueAllFeaturesInCurretPartition(self, currentPartition):
        informationValueAllFeatures = {}
        for featureName in currentPartition:
            if (featureName == self.classHeaderName):
                continue
            else:
                curFeatOptionInfoValue = self._calcInfoValueOnOptions(currentPartition, featureName)
                ivSum = 0
                for index in range(len(curFeatOptionInfoValue)):
                    curIV = curFeatOptionInfoValue[index]
                    ivSum = curIV + ivSum
                ivSum = -1*ivSum
                informationValueAllFeatures[featureName] = ivSum
        print('IV!!')
        print(informationValueAllFeatures)
        return informationValueAllFeatures
    
    def _calcInfoValueOnOptions(self, currentPartition, featureName):
        infoValueAllOptionsInCurFeature = []
        numberObservations = len(currentPartition.index)
        featureOptions = currentPartition[featureName].unique()
        for option in featureOptions:
            optionDF = currentPartition.loc[currentPartition[featureName] == option]
            infoValueAllOptionsInCurFeature.append(self._calcOptionInfoValue(optionDF, numberObservations))
        return infoValueAllOptionsInCurFeature  
    
    def _calcOptionInfoValue(self, optionDF, numberObservations):
        numberOfOccurancesOption = len(optionDF.index)
        infoValueOption = ((numberOfOccurancesOption/(numberObservations))* math.log2(numberOfOccurancesOption/(numberObservations)))
        return infoValueOption
      
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
        
        print('GAIN RATIO!!')
        print(gainRatioAllFeatures)
        return(gainRatioAllFeatures)
        
    def _calcExpectedEntropyAllFeaturesInCurrentParition(self, currentPartition):
        expectedEntropyAllFeatures = {}
        for featureName in currentPartition:
            if (featureName == self.classHeaderName):
                continue
            else:
                print(featureName)
                curFeatOptionsProb = self._calcProabilityOnOptions(currentPartition, featureName)
                curFeatOptionsEntropy = self._calcEntropyOnOptions(currentPartition, featureName)
                print(curFeatOptionsProb)
                print(curFeatOptionsEntropy)
                entropySum = 0 
                for index in range(len(curFeatOptionsProb)):
                    curProb = curFeatOptionsProb[index]
                    curEntrop = curFeatOptionsEntropy[index]
                    multTerm = curProb*curEntrop
                    entropySum = entropySum + multTerm
                expectedEntropyAllFeatures[featureName] = entropySum
        print(expectedEntropyAllFeatures)
        return expectedEntropyAllFeatures
            
    def _calcProabilityOnOptions(self, currentPartition, featureName):
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
    
    def _calcEntropyOnOptions(self, currentPartition, featureName):
        entropyAllOptionsInCurFeature = []
        featureOptions = currentPartition[featureName].unique()
        for option in featureOptions:
            optionDF = currentPartition.loc[currentPartition[featureName] == option]
            entropyAllOptionsInCurFeature.append(self._calcOptionEntropy(optionDF))
        return entropyAllOptionsInCurFeature
            
    def _calcOptionEntropy(self, optionDF):
        entropyI = 0
        if(self.numClassProblem == 2):
            classOptionCounts = optionDF[self.classHeaderName].value_counts()
            #Class 1 Option Name and Count
            opt1Name = classOptionCounts.index[0]
            opt1Count = classOptionCounts[opt1Name]

            #TODO: If Unique only returns 1 this will be a problem
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
    
    
    
    def _determineMaxGainRatioFeature(self, currentPartition):
        entPar = self._calcPartitionEntropy(currentPartition)
        expEnt = self._calcExpectedEntropyAllFeaturesInCurrentParition(currentPartition)
        gainPar = self._calcGainAllFeaturesInCurrentParition(entPar, expEnt)
        infoValPar = self._calcInformationValueAllFeaturesInCurretPartition(currentPartition)
        gainRatio = self._calGainRatioAllFeaturesInCurrentPartition(gainPar, infoValPar)
        maxGainRatioFeature = max(gainRatio, key=gainRatio.get)
        print('Feature with Max Gain Ratio: \t' + maxGainRatioFeature)
        return maxGainRatioFeature
    
    def _getSplitType(self, featureName):
        domainTypeDict = self.ID3AllDataSets[self.dataSetName].id3ColTypes
        splitType = domainTypeDict[featureName]
        return splitType
    

                
    def runID3Algo(self, inputDataset):
        print('Running ID3')
        self.generateTree(inputDataset, self.ID3DecTreeRoot)
        print('End of ID3 Tree Building')
        
    def generateTree(self, currentPartition, currentNode):
        #Check if the currentPartition only has one Class Label in it
        #if so return a leaf
        classCount = currentPartition[self.classHeaderName].value_counts()
        if(len(classCount) == 1):
            currentNode.setNodeContent(classCount.index[0])
            return
        
        #First Time throuhg the tree, the root node is None
        #Fill in that Root with the max GainRatio Feature from the Data Set 
        maxGRFeatureName = self._determineMaxGainRatioFeature(currentPartition)
        currentNode.setNodeContent(maxGRFeatureName)
        splitType = self._getSplitType(maxGRFeatureName)
        
        #Numeric Split 
        if (splitType == 'Num'):
            #TODO: Insert how to split based on this
            print('Not Yet Implemented')
            
        #Categorical Split
        elif(splitType == 'Cat'):
            #Determine the Range of the Feature 
            #Pull it's unique attributes
            featureOptions = currentPartition[maxGRFeatureName].unique()
            childIdx = 0;
            for option in featureOptions:
                currentNode.addChildNode(option)
                currentNode.childNodePathDict[option] = childIdx;
                print('Child Node is:' + option)
                
                #Want to build a new parition of that data that 
                #only includes instances where that Feature has the Attribue
                #that was added as the child 
                newPartition = currentPartition[(currentPartition[maxGRFeatureName] == option)]
                newBaseNode = currentNode.getChildNode(childIdx)
                childIdx = childIdx + 1;
                print('Debug Break point prior to entering recursive call')
                self.generateTree(newPartition, newBaseNode)
            
            

class Node:
    def __init__(self):
        self.nodeContent = None
        self.childrenNodes = []
        self.childNodePathDict = {}  #Key will be option name, value will be childIdx
        
    def setNodeContent(self, nodeLabel: str):
        self.nodeContent = nodeLabel;
        
    def getNodeContent(self):
        return self.nodeContent
    
    def addChildNode(self, inputNodeName):
        newChild = Node()
        newChild.setNodeContent(inputNodeName)
        self.childrenNodes.append(newChild)
    
    def getChildNode(self, inputNodeChildIdx):
        return self.childrenNodes[inputNodeChildIdx]
    