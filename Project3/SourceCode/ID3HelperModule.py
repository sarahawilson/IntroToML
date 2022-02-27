# Sarah Wilson 
# 303 - 921 - 7225
# Project 3
# Introduction to Machine Learning

import pandas as pd
import math

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
    
    def _calcExpectedEntropy(self, currentPartition):
        expectedEntropyE_Pi = None
        
        return expectedEntropyE_Pi
                
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