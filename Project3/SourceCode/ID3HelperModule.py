# Sarah Wilson 
# 303 - 921 - 7225
# Project 3
# Introduction to Machine Learning

import pandas as pd
import math
import numpy as np

class ID3Helper:
    def __init__(self, dataSetName: str, numClassProblem: int, classHeaderName: str):
        self.name = "ID3 Helper"
        self.treeRootNode = Node()
        self.numClassProblem = numClassProblem
        self.classHeaderName = classHeaderName
        self.dataSetName = dataSetName
        
    
    def _dropUniqueIDs(self, currentParition, dropHeaderName):
        print('')
    
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
        print(gainAllFeatures)
        return gainAllFeatures
    
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
    
                
    def runID3Algo(self, inputDataset):
        print('Running ID3')
        self.generateTree(inputDataSet)
        
    def generateTree(self, currentPartition):
        print('')

        


class Node:
    def __init__(self):
        self.leftNode = None
        self.rightNode = None
        self.nodeContent = None
        
    def setLeftNode(self):
        print('Setting the Left Node')
    
    def getLeftNode(self):
        print('Getting the Left Node')
        return self.leftNode
    
    def setRightNode(self):
        print('Setting the Right Node')
        
    def getRightNode(self):
        print('Getting the Right Node')
        return self.rightNode