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
    
    def runKFoldCrossVal_OnSingleDataSet_ForTuningKValKNN(self, toRunOnDataSetName):
        curDataFrameFoldList = self.create_folds(self.allDataSets[toRunOnDataSetName].finalData_Validation20PercentSet)
        curDataFramePredictor = self.allDataSets[toRunOnDataSetName].predictor
        curDataFrameTaskType = self.allDataSets[toRunOnDataSetName].taskType
        kNNValues = [1,3,5,7]
        
        allFoldMSE = []
        allFoldError = []
        
        
        for kVal in kNNValues:
            for iFoldIndex in range(self.numFolds):
                print('Tuning on Fold: \t' + str(iFoldIndex))
                print('Tuning for kNN Value: \t' + str(kVal))
                loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                tuneTestDF = loopDataFrameFoldList.pop(iFoldIndex)
                tuneTrainDF = pd.concat(loopDataFrameFoldList, axis=0)
                    
                #Insert the Step where the Algorithm Runs
                #Change out for the current algorithm being tested
                (curFoldMSE, curFoldErr) = self.algoHelper.runKNN_Algorithm(kVal, tuneTestDF, tuneTrainDF, curDataFramePredictor, curDataFrameTaskType)
                allFoldMSE.append(curFoldMSE)
                allFoldError.append(curFoldErr)
                print('-')
            
            #Calcualte the Average Accuracy and Error on the Folds
            if(allFoldMSE[0]!=None):
                avgMSE = sum(allFoldMSE)/(self.numFolds)
            else:
                avgMSE = None
            if(allFoldError[0]!=None):
                avgError = sum(allFoldError)/(self.numFolds)
            else:
                avgError = None
            print('Average Error on Fold: \t' + str(avgError))
            print('Average Mean Square Error on Fold: \t' + str(avgMSE))
            print('----')
            print('----')
            #Reset for next k Value
            allFoldMSE = []
            allFoldError = []


    def runKFoldCrossVal_OnSingleDataSet_ForTuningSigmaAndK(self, toRunOnDataSetName):
        curDataFrameFoldList = self.create_folds(self.allDataSets[toRunOnDataSetName].finalData_Validation20PercentSet)
        curDataFramePredictor = self.allDataSets[toRunOnDataSetName].predictor
        curDataFrameTaskType = self.allDataSets[toRunOnDataSetName].taskType
        kNNValues = [1,3,5,7]
        sigmaValues = [0.01,0.1,1,10]
        
        
        allFoldMSE = []
        allFoldError = []
        
        for sigmaVal in sigmaValues:
            for kVal in kNNValues:
                for iFoldIndex in range(self.numFolds):
                    print('Tuning on Fold: \t' + str(iFoldIndex))
                    print('Tuning for K Value: \t' + str(kVal))
                    print('Tuning for Sigma Value: \t' + str(sigmaVal))
                    loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                    tuneTestDF = loopDataFrameFoldList.pop(iFoldIndex)
                    tuneTrainDF = pd.concat(loopDataFrameFoldList, axis=0)
                    
                    #Insert the Step where the Algorithm Runs
                    #Change out for the current algorithm being tested
                    (curFoldMSE, curFoldErr) = self.algoHelper.runKNN_AlgorithmForRegression(kVal, tuneTestDF, tuneTrainDF, 
                                                                                             curDataFramePredictor, curDataFrameTaskType, sigmaVal)
                    allFoldMSE.append(curFoldMSE)
                    allFoldError.append(curFoldErr)
                    print('-')
            
                #Calcualte the Average Accuracy and Error on the Folds
                if(allFoldMSE[0]!=None):
                    avgMSE = sum(allFoldMSE)/(self.numFolds)
                else:
                    avgMSE = None
                if(allFoldError[0]!=None):
                    avgError = sum(allFoldError)/(self.numFolds)
                else:
                    avgError = None
                    print('Average Error on Fold: \t' + str(avgError))
                    print('Average Mean Square Error on Fold: \t' + str(avgMSE))
                    print('----')
                    print('----')
                    #Reset for next k Value
                    allFoldMSE = []
                    allFoldError = []        

    def runKFoldCrossVal_OnSingleDataSet_ForExp(self, toRunOnDataSetName, optK, optSigma = None):
        curDataFrameFoldList = self.create_folds(self.allDataSets[toRunOnDataSetName].finalData_ExperimentSet)
        curDataFramePredictor = self.allDataSets[toRunOnDataSetName].predictor
        curDataFrameTaskType = self.allDataSets[toRunOnDataSetName].taskType

        allFoldAccuracy = []
        allFoldError = []
        
        print('START OF EXPRIMENT FOR:')
        print(toRunOnDataSetName + 'Data Set')
        for iFoldIndex in range(self.numFolds):
            print('Running on Fold: \t' + str(iFoldIndex))
            loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
            runTestDF = loopDataFrameFoldList.pop(iFoldIndex)
            runTrainDF = pd.concat(loopDataFrameFoldList, axis=0)
                    
            #Insert the Step where the Algorithm Runs
            #Change out for the current algorithm being tested
            if(curDataFrameTaskType == 'Classification'):
                (curFoldAcc, curFoldErr) = self.algoHelper.runKNN_Algorithm(optK, runTestDF, runTrainDF, curDataFramePredictor, curDataFrameTaskType)
            elif(curDataFrameTaskType == 'Regression'):
                (curFoldAcc, curFoldErr) = self.algoHelper.runKNN_AlgorithmForRegression(optK, runTestDF, runTrainDF, curDataFramePredictor, curDataFrameTaskType, optSigma)
            allFoldAccuracy.append(curFoldAcc)
            allFoldError.append(curFoldErr)
            print('-')
            
        #Calcualte the Average Accuracy and Error on the Folds
        if(allFoldAccuracy[0]!=None):
            avgAcc = sum(allFoldAccuracy)/(self.numFolds)
        else:
            avgAcc = None
        if(allFoldError[0]!=None):
            avgError = sum(allFoldError)/(self.numFolds)
        else:
            avgError = None
        print('Average Error on Fold: \t' + str(avgError))
        print('Average Root Square Mean Error on Fold: \t' + str(avgAcc))
        print('----')
        print('----')
        
    