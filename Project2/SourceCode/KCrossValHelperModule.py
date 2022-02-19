# Sarah Wilson 
# 303 - 921 - 7225
# Project 2
# Introduction to Machine Learning

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import copy
import KNNAlgoHelperModule

class KCrossValHelper:
    def __init__(self,
                 allDataSets: Dict,
                 ):
        
        self.name = 'KCrossVal'
        self.numFolds = 5
        self.allDataSets = allDataSets
        self.algoHelper = KNNAlgoHelperModule.KNNAlgoHelper(self.allDataSets)
        
        
    def createValidation_TuneAndExperimentSets(self, runOn = "AllDataSets"):
    # Splits the overall data sets into the 20% that is needed for Validation
    # The other 80% is left for the full algorithm experiment 
    
        if(runOn == "AllDataSets"):
            for dataSetName in self.allDataSets: 
                tempCurDataSet = self.allDataSets[dataSetName].finalData.copy(deep=True)
                #Create the 20% Set
                self.allDataSets[dataSetName].finalData_Validation20PercentSet = tempCurDataSet.sample(frac=0.2, random_state=1)
            
                #Create the 80% Set 
                self.allDataSets[dataSetName].finalData_ExperimentSet = tempCurDataSet.drop(self.allDataSets[dataSetName].finalData_Validation20PercentSet.index)
            
    def create_folds(self, inputDataFrame):
        # Input data frame
        #   and the number of folds (int) to create out of this data frame
        # Creates the number of disjoint folds from the input data frame
        # Returns a list of these dataframes
        tempInputDataFrame = inputDataFrame.copy(deep=True)
        kFoldDataFramesTest = np.array_split(tempInputDataFrame, self.numFolds)

        return kFoldDataFramesTest; 
    
    def runKFoldCrossVal_OnSingleDataSet_ForTuning(self, toRunOnDataSetName):
        curDataFrameFoldList = self.create_folds(self.allDataSets[toRunOnDataSetName].finalData_Validation20PercentSet)
        curDataFramePredictor = self.allDataSets[toRunOnDataSetName].predictor
        curDataFrameTaskType = self.allDataSets[toRunOnDataSetName].taskType
        kvals = [1,3,5,7]
        
        for iFoldIndex in range(self.numFolds):
            loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
            tuneTestDF = loopDataFrameFoldList.pop(iFoldIndex)
            tuneTrainDF = pd.concat(loopDataFrameFoldList, axis=0)
                    
            #Insert the Step where the Algorithm Runs
            self.algoHelper.runKNNAlgorithm(kvals, tuneTestDF, tuneTrainDF, curDataFramePredictor, curDataFrameTaskType)
        
        
    def runKFoldCrossVal_OnAllDataSets_ForTuning(self, runOn = 'AllDataSets'):
        #print(printMessage)
        #errorPerFold = [] #Error = Number Wrong / Total
        #accuracyPerFold = [] #Accuracy = Number Right / Total
        
        if(runOn == "AllDataSets"):
            for dataSetName in self.allDataSets: 
                #Create the Folds
                curDataFrameFoldList = self.create_folds(self.allDataSets[dataSetName].finalData_Validation20PercentSet)
                
                for iFoldIndex in range(self.numFolds):
                    loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                    tuneTestDF = loopDataFrameFoldList.pop(iFoldIndex)
                    tuneTrainDF = pd.concat(loopDataFrameFoldList, axis=0)
                    
                    #Insert the Step where the Algorithm Runs
                    self.algoHelper.runKNNAlgorithm(tuneTestDF, tuneTrainDF)
        
    