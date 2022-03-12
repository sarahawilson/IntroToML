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
        #self.runTestDFThroughTree(testDF)
        #return self.ID3DecTreeRoot
        
    def generateTree(self, currentPartition, currentNode):
        #Check if the currentPartition only has one Class Label in it
        #if so return a leaf
        classCount = currentPartition[self.classHeaderName].value_counts()
        if(len(classCount) == 1):
            currentNode.setNodeContent(classCount.index[0])
            return
        
        #Determine the maximum gain ratio feature 
        maxGRFeatureName = self._determineMaxGainRatioFeature(currentPartition)
        currentNode.setNodeContent(maxGRFeatureName)
        featureType = self._getFeatureType(maxGRFeatureName)
        
        #Numeric Feature 
        if (featureType == 'Num'):
            sortedPartition = currentPartition.sort_values(by=[maxGRFeatureName])
            sortedPartition['Count Class Change'] = sortedPartition[self.classHeaderName].ne(sortedPartition[self.classHeaderName].shift()).cumsum()
            sortedPartition['Change Occured'] = sortedPartition['Count Class Change'].diff()
            sortedPartition['Before Change Occured'] = 0
            sortedPartition['Before Change Occured'][:-1] = sortedPartition['Change Occured'][1:]
            beforeChangeDF = sortedPartition[sortedPartition['Before Change Occured']==1]
            possSplitVals = []
            for rowIdx in range(len(beforeChangeDF)):
                possSplitVals.append(beforeChangeDF[maxGRFeatureName].values[rowIdx])
                                        
            childIdx = 0
            for splitVal in possSplitVals:
                stringVal = str(splitVal)
                pathName = "<=" + stringVal
                currentNode.addChildNode()
                currentNode.childNodePathDict[pathName] = childIdx
                
                #Want to build a new parition of that data that 
                #only includes instances where that Feature has the Attribue less  than or equal to
                #the split value 
                newPartition = currentPartition[(currentPartition[maxGRFeatureName] <= splitVal)]
                if(len(newPartition) == 0):
                    print('WHOA WHOA WHOA')
                    print('something is not right')
                newBaseNode = currentNode.getChildNode(childIdx)
                childIdx = childIdx + 1;
                print('Debug Break point prior to entering recursive call')
                self.generateTree(newPartition, newBaseNode)
                
                    
        #Categorical Feature
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
              
        
    def dropUniqueIDs(self, dataFrame):
        if(self.dropHeaderName != None):
            resultDF = dataFrame.drop([self.dropHeaderName], axis=1)
        return resultDF
    
    def _calcPartitionEntropy(self, currentPartition):
        entropyI_Pi = None
        if(self.numClassProblem == 2):
            classOptions = currentPartition[self.classHeaderName].unique()
            #print(classOptions)
            classOptionCounts = currentPartition[self.classHeaderName].value_counts()
            print(classOptionCounts)
            if(classOptionCounts.size == 1):
                test = 2
            #print(classOptionCounts)
            #Class 1 Option Name and Count
            opt1Name = classOptionCounts.index[0]
            opt1Count = classOptionCounts[opt1Name]
            
            #Class 2 Option Name and Count
            opt2Name = classOptionCounts.index[1]
            opt2Count = classOptionCounts[opt2Name]
            
            termOpt1 = ((-opt1Count/(opt1Count + opt2Count))* math.log2(opt1Count/(opt1Count + opt2Count)))
            termOpt2 = ((opt2Count/(opt1Count + opt2Count))* math.log2(opt2Count/(opt1Count + opt2Count)))
            
            entropyI_Pi = termOpt1 - termOpt2
            #print(entropyI_Pi)
            
        return entropyI_Pi 
    
    def _calcGainAllFeaturesInCurrentParition(self, entropyOfPartition, expectedEntropyAllFeatures: dict):
        #TODO: Add comments
        gainAllFeatures = {}
        for featureName in expectedEntropyAllFeatures:
            featureExpectedEntropy = expectedEntropyAllFeatures[featureName]
            gainAllFeatures[featureName] = entropyOfPartition - featureExpectedEntropy
        #print('GAIN!!')
        #print(gainAllFeatures)
        return gainAllFeatures
    
    def _calcInformationValueAllFeaturesInCurretPartition(self, currentPartition):
        informationValueAllFeatures = {}
        for featureName in currentPartition:
            if (featureName == self.classHeaderName):
                continue
            else:
                featureType = self._getFeatureType(featureName)
                if(featureType == 'Cat'):
                    curFeatOptionInfoValue = self._calcInfoValueOnOptions(currentPartition, featureName)
                if(featureType == 'Num'):
                    curFeatOptionInfoValue = self._calcInfoValueOnNumericOptions(currentPartition, featureName)
                      
                ivSum = 0
                for index in range(len(curFeatOptionInfoValue)):
                    curIV = curFeatOptionInfoValue[index]
                    ivSum = curIV + ivSum
                ivSum = -1*ivSum
                informationValueAllFeatures[featureName] = ivSum
        #print('IV!!')
        #print(informationValueAllFeatures)
        return informationValueAllFeatures
    
    def _calcInfoValueOnOptions(self, currentPartition, featureName):
        infoValueAllOptionsInCurFeature = []
        numberObservations = len(currentPartition.index)
        featureOptions = currentPartition[featureName].unique()
        for option in featureOptions:
            optionDF = currentPartition.loc[currentPartition[featureName] == option]
            infoValueAllOptionsInCurFeature.append(self._calcOptionInfoValue(optionDF, numberObservations))
        return infoValueAllOptionsInCurFeature  
    
    def _calcInfoValueOnNumericOptions(self, currentPartition, featureName):
        infoValueAllSplitsInCurFeature = []
        numberObservations = len(currentPartition.index)
        sortedPartition = currentPartition.sort_values(by=[featureName])
        sortedPartition['Count Class Change'] = sortedPartition[self.classHeaderName].ne(sortedPartition[self.classHeaderName].shift()).cumsum()
        sortedPartition['Change Occured'] = sortedPartition['Count Class Change'].diff()
        sortedPartition['Before Change Occured'] = 0
        sortedPartition['Before Change Occured'][:-1] = sortedPartition['Change Occured'][1:]
        beforeChangeDF = sortedPartition[sortedPartition['Before Change Occured']==1]
        possSplitVals = []
        for rowIdx in range(len(beforeChangeDF)):
            possSplitVals.append(beforeChangeDF[featureName].values[rowIdx])
            
        for splitVal in possSplitVals:
            lessThanEqToDF = sortedPartition[sortedPartition[featureName] <= splitVal]
            if(self._calcOptionInfoValue(lessThanEqToDF, numberObservations) == 0):
                print('DB')
            infoValueAllSplitsInCurFeature.append(self._calcOptionInfoValue(lessThanEqToDF, numberObservations))
        return infoValueAllSplitsInCurFeature
    
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
        
        #print('GAIN RATIO!!')
        #print(gainRatioAllFeatures)
        return(gainRatioAllFeatures)
        
    def _calcExpectedEntropyAllFeaturesInCurrentParition(self, currentPartition):
        expectedEntropyAllFeatures = {}
        for featureName in currentPartition:
            if (featureName == self.classHeaderName):
                continue
            else:
                featureType = self._getFeatureType(featureName)
                #print(featureName)
                if(featureType == 'Cat'):
                    curFeatOptionsProb = self._calcProabilityOnOptions(currentPartition, featureName)
                    curFeatOptionsEntropy = self._calcEntropyOnOptions(currentPartition, featureName)
                if(featureType == 'Num'):
                    curFeatOptionsProb = self._calcProabilityOnNumericOptions(currentPartition, featureName)
                    curFeatOptionsEntropy = self._calcEntropyOnNumericOptions(currentPartition, featureName)
                #print(curFeatOptionsProb)
                #print(curFeatOptionsEntropy)
                entropySum = 0 
                for index in range(len(curFeatOptionsProb)):
                    curProb = curFeatOptionsProb[index]
                    curEntrop = curFeatOptionsEntropy[index]
                    multTerm = curProb*curEntrop
                    entropySum = entropySum + multTerm
                expectedEntropyAllFeatures[featureName] = entropySum
        #print(expectedEntropyAllFeatures)
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
    
    def _calcProabilityOnNumericOptions(self, currentPartition, featureName):
        probsAllSplitsInCurFeature = []
        numberObservations = len(currentPartition.index)
        sortedPartition = currentPartition.sort_values(by=[featureName])
        sortedPartition['Count Class Change'] = sortedPartition[self.classHeaderName].ne(sortedPartition[self.classHeaderName].shift()).cumsum()
        sortedPartition['Change Occured'] = sortedPartition['Count Class Change'].diff()
        sortedPartition['Before Change Occured'] = 0
        sortedPartition['Before Change Occured'][:-1] = sortedPartition['Change Occured'][1:]
        beforeChangeDF = sortedPartition[sortedPartition['Before Change Occured']==1]
        possSplitVals = []
        for rowIdx in range(len(beforeChangeDF)):
            possSplitVals.append(beforeChangeDF[featureName].values[rowIdx])
        
        uniquePossSplits = np.unique(np.array(possSplitVals))
        uniquePossSplitsList = uniquePossSplits.tolist()
        
        for splitVal in uniquePossSplitsList:
            lessThanEqToDF = sortedPartition[sortedPartition[featureName] <= splitVal]
            optionCount = len(lessThanEqToDF)
            probibilitySplit = optionCount / numberObservations
            probsAllSplitsInCurFeature.append(probibilitySplit)
        return probsAllSplitsInCurFeature
            
    
    def _calcEntropyOnOptions(self, currentPartition, featureName):
        entropyAllOptionsInCurFeature = []
        featureOptions = currentPartition[featureName].unique()
        for option in featureOptions:
            optionDF = currentPartition.loc[currentPartition[featureName] == option]
            entropyAllOptionsInCurFeature.append(self._calcOptionEntropy(optionDF))
        return entropyAllOptionsInCurFeature
    
    def _calcEntropyOnNumericOptions(self, currentPartition, featureName):
        entropyAllSplitsInCurFeature = []
        sortedPartition = currentPartition.sort_values(by=[featureName])
        sortedPartition['Count Class Change'] = sortedPartition[self.classHeaderName].ne(sortedPartition[self.classHeaderName].shift()).cumsum()
        sortedPartition['Change Occured'] = sortedPartition['Count Class Change'].diff()
        sortedPartition['Before Change Occured'] = 0
        sortedPartition['Before Change Occured'][:-1] = sortedPartition['Change Occured'][1:]
        beforeChangeDF = sortedPartition[sortedPartition['Before Change Occured']==1]
        possSplitVals = []
        for rowIdx in range(len(beforeChangeDF)):
            possSplitVals.append(beforeChangeDF[featureName].values[rowIdx])
            
        for splitVal in possSplitVals:
            lessThanEqToDF = sortedPartition[sortedPartition[featureName] <= splitVal]
            entropyAllSplitsInCurFeature.append(self._calcOptionEntropy(lessThanEqToDF))

        return entropyAllSplitsInCurFeature
            
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
        
        entPar = self._calcPartitionEntropy(currentPartition) #Same for Num/Cat Data
        expEnt = self._calcExpectedEntropyAllFeaturesInCurrentParition(currentPartition)
        gainPar = self._calcGainAllFeaturesInCurrentParition(entPar, expEnt)
        
        infoValPar = self._calcInformationValueAllFeaturesInCurretPartition(currentPartition)
        gainRatio = self._calGainRatioAllFeaturesInCurrentPartition(gainPar, infoValPar)
        maxGainRatioFeature = max(gainRatio, key=gainRatio.get)
        print('Feature with Max Gain Ratio: \t' + maxGainRatioFeature)
        return maxGainRatioFeature
    
    def _getFeatureType(self, featureName):
        domainTypeDict = self.ID3AllDataSets[self.dataSetName].id3ColTypes
        featureType = domainTypeDict[featureName]
        return featureType
                    
    def runTestDFThroughTree(self, testDF):
        for rowIdx in range(len(testDF)):
            self.passObservationThroughTree(testDF, rowIdx, self.ID3DecTreeRoot)
            
        
    def passObservationThroughTree(self, testDF, rowIdx, currentNode):
        curNodeFeature = currentNode.getNodeContent()
        featureType = self._getFeatureType(curNodeFeature)
        
        #Have run into a leaf node stop
        numChildren = len(currentNode.childrenNodes)
        if(numChildren == 0):
            return 
        
    

            

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
    