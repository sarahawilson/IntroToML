# Sarah Wilson 
# 303 - 921 - 7225
# Project 3
# Introduction to Machine Learning

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import copy
import ID3HelperModule
import CARTHelperModule

class KCrossValHelper:
    def __init__(self,
                 allDataSets: Dict,
                 ):
        
        self.name = 'KCrossVal'
        self.numFolds = 5
        self.allDataSets = allDataSets
        
        #self._createValidation_TuneAndExperimentSets()
        self.ID3TreeFoldDict = {}
        self.CARTTreeFoldDict = {}
        
        
    def _createValidation_TuneAndExperimentSets(self, runOn = "AllDataSets"):
    # Splits the overall data sets into the 20% that is needed for Validation
    # The other 80% is left for the full algorithm experiment 
    
        if(runOn == "AllDataSets"):
            for dataSetName in self.allDataSets: 
                tempCurDataSet = self.allDataSets[dataSetName].finalData.copy(deep=True)
                #Create the 20% Set
                self.allDataSets[dataSetName].finalData_Validation20PercentSet = tempCurDataSet.sample(frac=0.2, random_state=1)
            
                #Create the 80% Set 
                self.allDataSets[dataSetName].finalData_ExperimentSet = tempCurDataSet.drop(self.allDataSets[dataSetName].finalData_Validation20PercentSet.index)
            
    def _createValidation_TuneAndExperimentSetsAfterDropUnique(self, dataSetName, dropedDF):
        # Splits the overall data sets into the 20% that is needed for Validation
       # The other 80% is left for the full algorithm experiment 
    
        #Create the 20% Set
        self.allDataSets[dataSetName].finalData_Validation20PercentSet = dropedDF.sample(frac=0.2, random_state=1)
            
        #Create the 80% Set 
        self.allDataSets[dataSetName].finalData_ExperimentSet = dropedDF.drop(self.allDataSets[dataSetName].finalData_Validation20PercentSet.index)
            
    def _create_folds(self, inputDataFrame):
        # Input data frame
        #   and the number of folds (int) to create out of this data frame
        # Creates the number of disjoint folds from the input data frame
        # Returns a list of these dataframes
        tempInputDataFrame = inputDataFrame.copy(deep=True)
        kFoldDataFramesTest = np.array_split(tempInputDataFrame, self.numFolds)

        return kFoldDataFramesTest; 
    
    def cal_mean_std(self, inputDataFrame, colHeaders):
        #Calcualtes the mean and stardand deviation on a certain set of columns 
        # inputDataFrame is the pd.dataframe to get the data from
        # colApply is a list of columns to get the data from in the inputDataFrame
        mean_stdList = []
   
        for col in colHeaders:
            meanCol = inputDataFrame[col].mean(axis=0, skipna=True)
            stdCol = inputDataFrame[col].std(axis=0, skipna=True)
            #If statement here to adddress the divide by zero error in Z Standardization
            if(stdCol == 0.0):
                stdCol = 1.0
        
            meanStdTuple = (col, meanCol, stdCol)
            mean_stdList.append(meanStdTuple)
        return mean_stdList

    def zStanderdize_data(self, inputDataFrame, mean_stdList):
        #Applies the Z Standardization to the input data set.
        #On the columns indcaited in the list mean_stdList
        manipulateInputDataFrame = inputDataFrame.copy(deep=True)
        for element in mean_stdList:
            curCol = element[0]
            meanCol = element[1]
            stdCol = element[2]
            manipulateInputDataFrame = manipulateInputDataFrame.apply(lambda dfCol: self.zStanderdize_ApplyFunction(dfCol, meanCol, stdCol) if dfCol.name == curCol else dfCol)
        
        return manipulateInputDataFrame

    def zStanderdize_ApplyFunction(self, inputDataVaule, mean, std):
        #Describes the z-Standerdize funtion
        zStand = (inputDataVaule - mean)/std;
        return zStand       
    


    def runKFoldCrossVal_ID3_Univariate(self, dataSetName, predictorName, numClassProb, dropLabel = None):
        id3_Helper = ID3HelperModule.ID3Helper(self.allDataSets[dataSetName].name, numClassProb, predictorName, dropLabel, self.allDataSets)
        if(dropLabel != None):
            finalID3Data = id3_Helper.dropUniqueIDs(self.allDataSets[dataSetName].finalData)
        else:
            finalID3Data = self.allDataSets[dataSetName].finalData
        
        #Sanity Check on Tree Building - For Simple Data Sets
        #self.ID3Tree = id3_Helper.runID3Algo(None, finalID3Data)
        
        self._createValidation_TuneAndExperimentSetsAfterDropUnique(dataSetName, finalID3Data)
        
        curDataFrameFoldList = self._create_folds(self.allDataSets[dataSetName].finalData_ExperimentSet)
        
        for iFoldIndex in range(self.numFolds):
            print('Fold:' + str(iFoldIndex))
            loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
            testDF = loopDataFrameFoldList.pop(iFoldIndex)
            trainDF = pd.concat(loopDataFrameFoldList, axis=0)    
            
            curFoldID3Tree = id3_Helper.runID3Algo(testDF, trainDF)
            foldName = 'Fold' + str(iFoldIndex)
            self.ID3TreeFoldDict[foldName] = curFoldID3Tree
            #Clear the Tree for the Next time through
            id3_Helper.clearTree()
    
    def runKFoldCrossVal_CART_Univariate(self, dataSetName, predictorName, dropLabel = None):
        cart_Helper = CARTHelperModule.CARTHelper(self.allDataSets[dataSetName].name, predictorName, dropLabel, self.allDataSets)
            
        if(dataSetName == 'Albalone'):
            self.allDataSets[dataSetName].applyOneHotEncoding(['Sex'])
        elif (dataSetName == 'Forest Fire'):
            self.allDataSets[dataSetName].applyOneHotEncoding(['month', 'day'])
            #self.allDataSets[dataSetName].finalData = self.allDataSets[dataSetName].finalData.drop(['month', 'day'], axis =1)
        elif (dataSetName == 'Computer Hardware'):
            self.allDataSets[dataSetName].applyOneHotEncoding(['Vendor Name', 'Model Name'])
            #self.allDataSets[dataSetName].finalData = self.allDataSets[dataSetName].finalData.drop(['month', 'day'], axis =1)
            
        if(dropLabel != None):
            finalCARTData = cart_Helper.dropUniqueIDs(self.allDataSets[dataSetName].finalData)
        else:
            finalCARTData = self.allDataSets[dataSetName].finalData
        
        #Sanity Check on Tree Building - For Simple Data Sets
        #self.ID3Tree = id3_Helper.runID3Algo(None, finalID3Data)
        
        self._createValidation_TuneAndExperimentSetsAfterDropUnique(dataSetName, finalCARTData)
        
        curDataFrameFoldList = self._create_folds(self.allDataSets[dataSetName].finalData_ExperimentSet)
        
        for iFoldIndex in range(self.numFolds):
            print('Fold:' + str(iFoldIndex))
            loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
            testDF = loopDataFrameFoldList.pop(iFoldIndex)
            trainDF = pd.concat(loopDataFrameFoldList, axis=0)    
            
            curFoldID3Tree = cart_Helper.runCARTAlgo(testDF, trainDF)
            foldName = 'Fold' + str(iFoldIndex)
            self.CARTTreeFoldDict[foldName] = curFoldID3Tree
            #Clear the Tree for the Next time through
            cart_Helper.clearTree()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    