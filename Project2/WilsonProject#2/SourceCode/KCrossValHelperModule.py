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
        
        self._createValidation_TuneAndExperimentSets()
        self.algoHelper = KNNAlgoHelperModule.KNNAlgoHelper()
        
        
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
    
    def runKFoldCrossVal_ForNormalKNN_Tuning(self, toRunOnDataSetName: str, kVals: list, sigmaVals: list, zStand = False, zStandHeaders = None):
        curDataFrameFoldList = self._create_folds(self.allDataSets[toRunOnDataSetName].finalData_Validation20PercentSet)
        curDataFramePredictor = self.allDataSets[toRunOnDataSetName].predictor
        curDataFrameTaskType = self.allDataSets[toRunOnDataSetName].taskType
        
        allFoldRMSError = []
        allFoldClassificationError = []
        zStandTestTrainDict = {'Train Set': None, 'Test Set': None}
        
        if(curDataFrameTaskType == 'Regression'):
            print('---NORMAL KNN---')
            print('Tuning K and Sigma On: ' + toRunOnDataSetName)
            for kVal in kVals:
                print('K:' + str(kVal))
                for sigmaVal in sigmaVals:
                    print('Sigma:' + str(sigmaVal))
                    for iFoldIndex in range(self.numFolds):
                        print('Fold:' + str(iFoldIndex))
                        loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                        testDF = loopDataFrameFoldList.pop(iFoldIndex)
                        trainDF = pd.concat(loopDataFrameFoldList, axis=0)
            
                        zStandTestTrainDict['Test Set'] = testDF
                        zStandTestTrainDict['Train Set'] = trainDF
            
                        if(zStand):
                            meanStdTuple = self.cal_mean_std(zStandTestTrainDict['Train Set'], zStandHeaders)
                            testdf_zStandTrainSet = self.zStanderdize_data(zStandTestTrainDict['Train Set'], meanStdTuple)
                            testdf_zStandTestSet = self.zStanderdize_data(zStandTestTrainDict['Test Set'], meanStdTuple)
                
                            testDF = testdf_zStandTrainSet
                            testDF = testdf_zStandTestSet
    
    
                        #INSERT RUN NORMAL KNN
                        (curFoldRMSE, curFoldClassErr) = self.algoHelper.RunNormalKNN(kVal, sigmaVal, testDF, trainDF, curDataFramePredictor, curDataFrameTaskType)
                        allFoldRMSError.append(curFoldRMSE)
    
                    #Determine the overall Error for the Folds
                    avgAllFoldError= sum(allFoldRMSError)/(self.numFolds)
                    print('Average Root Mean Square Error on Fold: \t' + str(avgAllFoldError))
                    #Reset for next k/sigma Value
                    allFoldRMSError = []
                    
                        
        if(curDataFrameTaskType == 'Classification'):
            print('---NORMAL KNN---')
            print('Tuning K: ' + toRunOnDataSetName)
            for kVal in kVals:
                print('K:' + str(kVal))
                for iFoldIndex in range(self.numFolds):
                    loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                    testDF = loopDataFrameFoldList.pop(iFoldIndex)
                    trainDF = pd.concat(loopDataFrameFoldList, axis=0)
                    
                    (curFoldRMSE, curFoldClassErr) = self.algoHelper.RunNormalKNN(kVal, None, testDF, trainDF, curDataFramePredictor, curDataFrameTaskType)
                    allFoldClassificationError.append(curFoldClassErr)
                    
                    
                #Determine the overall Error for the Folds
                avgAllFoldError= sum(allFoldClassificationError)/(self.numFolds)
                print('Average Classification Error on Fold: \t' + str(avgAllFoldError))
                #Reset for next k value
                allFoldClassificationError = []
    

    def runKFoldCrossVal_NormalKNN(self, toRunOnDataSetName: str, kVal, sigmaVal, zStand = False, zStandHeaders = None):
        curDataFrameFoldList = self._create_folds(self.allDataSets[toRunOnDataSetName].finalData_ExperimentSet)
        curDataFramePredictor = self.allDataSets[toRunOnDataSetName].predictor
        curDataFrameTaskType = self.allDataSets[toRunOnDataSetName].taskType
        
        allFoldRMSError = []
        allFoldClassificationError = []
        zStandTestTrainDict = {'Train Set': None, 'Test Set': None}
        
        if(curDataFrameTaskType == 'Regression'):
            print('---NORMAL KNN---')
            print('Full Exp:')
            print('K:' + str(kVal))
            print('Sigma:' + str(sigmaVal))
            for iFoldIndex in range(self.numFolds):
                print('Fold:' + str(iFoldIndex))
                loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                testDF = loopDataFrameFoldList.pop(iFoldIndex)
                trainDF = pd.concat(loopDataFrameFoldList, axis=0)
            
                zStandTestTrainDict['Test Set'] = testDF
                zStandTestTrainDict['Train Set'] = trainDF
            
                if(zStand):
                    meanStdTuple = self.cal_mean_std(zStandTestTrainDict['Train Set'], zStandHeaders)
                    testdf_zStandTrainSet = self.zStanderdize_data(zStandTestTrainDict['Train Set'], meanStdTuple)
                    testdf_zStandTestSet = self.zStanderdize_data(zStandTestTrainDict['Test Set'], meanStdTuple)
                
                    testDF = testdf_zStandTrainSet
                    testDF = testdf_zStandTestSet
    
    
                #INSERT RUN NORMAL KNN
                (curFoldRMSE, curFoldClassErr) = self.algoHelper.RunNormalKNN(kVal, sigmaVal, testDF, trainDF, curDataFramePredictor, curDataFrameTaskType)
                allFoldRMSError.append(curFoldRMSE)
    
            #Determine the overall Error for the Folds
            avgAllFoldError= sum(allFoldRMSError)/(self.numFolds)
            print('Average Root Mean Square Error on Fold: \t' + str(avgAllFoldError))
            #Reset for next k/sigma Value
            allFoldRMSError = []
                    
                        
        if(curDataFrameTaskType == 'Classification'):
            print('---NORMAL KNN---')
            print('Full Exp:')
            print('K:' + str(kVal))
            for iFoldIndex in range(self.numFolds):
                loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                testDF = loopDataFrameFoldList.pop(iFoldIndex)
                trainDF = pd.concat(loopDataFrameFoldList, axis=0)
                    
                (curFoldRMSE, curFoldClassErr) = self.algoHelper.RunNormalKNN(kVal, None, testDF, trainDF, curDataFramePredictor, curDataFrameTaskType)
                allFoldClassificationError.append(curFoldClassErr)
                    
                    
            #Determine the overall Error for the Folds
            avgAllFoldError= sum(allFoldClassificationError)/(self.numFolds)
            print('Average Classification Error on Fold: \t' + str(avgAllFoldError))
            #Reset for next k value
            allFoldClassificationError = []    
 

    def runKFoldCrossVal_EditedKNN_Tune(self, toRunOnDataSetName: str, kVal, sigmaVal, epsilonVals: list, zStand = False, zStandHeaders = None):
        curDataFrameFoldList = self._create_folds(self.allDataSets[toRunOnDataSetName].finalData_Validation20PercentSet)
        curDataFramePredictor = self.allDataSets[toRunOnDataSetName].predictor
        curDataFrameTaskType = self.allDataSets[toRunOnDataSetName].taskType
        
        allFoldRMSError = []
        zStandTestTrainDict = {'Train Set': None, 'Test Set': None}
        
        if(curDataFrameTaskType == 'Regression'):
            print('---EDITED KNN---')
            print('Tuning for Epsilon')
            print('K:' + str(kVal))
            print('Sigma:' + str(sigmaVal))
            for epsilon in epsilonVals:
                print('Epsilon:' + str(epsilon))
                for iFoldIndex in range(self.numFolds):
                    print('Fold:' + str(iFoldIndex))
                    loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                    testDF = loopDataFrameFoldList.pop(iFoldIndex)
                    trainDF = pd.concat(loopDataFrameFoldList, axis=0)
            
                    zStandTestTrainDict['Test Set'] = testDF
                    zStandTestTrainDict['Train Set'] = trainDF
            
                    if(zStand):
                        meanStdTuple = self.cal_mean_std(zStandTestTrainDict['Train Set'], zStandHeaders)
                        testdf_zStandTrainSet = self.zStanderdize_data(zStandTestTrainDict['Train Set'], meanStdTuple)
                        testdf_zStandTestSet = self.zStanderdize_data(zStandTestTrainDict['Test Set'], meanStdTuple)
                
                        testDF = testdf_zStandTrainSet
                        testDF = testdf_zStandTestSet
    
    
                    #INSERT RUN NORMAL KNN
                    (curFoldRMSE, curFoldClassErr) = self.algoHelper.runEditedKNN(kVal, sigmaVal, epsilon, testDF, trainDF, curDataFramePredictor, curDataFrameTaskType)
                    allFoldRMSError.append(curFoldRMSE)
    
                #Determine the overall Error for the Folds
                avgAllFoldError= sum(allFoldRMSError)/(self.numFolds)
                print('Average Root Mean Square Error on Fold: \t' + str(avgAllFoldError))
                #Reset for next k/sigma Value
                allFoldRMSError = []
                    
    
    def runKFoldCrossVal_EditedKNN(self, toRunOnDataSetName: str, kVal, sigmaVal, epsilon, zStand = False, zStandHeaders = None):
        curDataFrameFoldList = self._create_folds(self.allDataSets[toRunOnDataSetName].finalData_ExperimentSet)
        curDataFramePredictor = self.allDataSets[toRunOnDataSetName].predictor
        curDataFrameTaskType = self.allDataSets[toRunOnDataSetName].taskType
        
        allFoldRMSError = []
        allFoldClassificationError = []
        zStandTestTrainDict = {'Train Set': None, 'Test Set': None}
        
        if(curDataFrameTaskType == 'Regression'):
            print('---EDITED KNN---')
            print('K:' + str(kVal))
            print('Sigma:' + str(sigmaVal))
            print('Epsilon:' + str(epsilon))
            for iFoldIndex in range(self.numFolds):
                print('Fold:' + str(iFoldIndex))
                loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                testDF = loopDataFrameFoldList.pop(iFoldIndex)
                trainDF = pd.concat(loopDataFrameFoldList, axis=0)
            
                zStandTestTrainDict['Test Set'] = testDF
                zStandTestTrainDict['Train Set'] = trainDF
            
                if(zStand):
                    meanStdTuple = self.cal_mean_std(zStandTestTrainDict['Train Set'], zStandHeaders)
                    testdf_zStandTrainSet = self.zStanderdize_data(zStandTestTrainDict['Train Set'], meanStdTuple)
                    testdf_zStandTestSet = self.zStanderdize_data(zStandTestTrainDict['Test Set'], meanStdTuple)
                    testDF = testdf_zStandTrainSet
                    testDF = testdf_zStandTestSet

                #Run Edited KNN
                (curFoldRMSE, curFoldClassErr) = self.algoHelper.runEditedKNN(kVal, sigmaVal, epsilon, testDF, trainDF, curDataFramePredictor, curDataFrameTaskType)
                allFoldRMSError.append(curFoldRMSE)
    
            #Determine the overall Error for the Folds
            avgAllFoldError= sum(allFoldRMSError)/(self.numFolds)
            print('Average Root Mean Square Error on Folds: \t' + str(avgAllFoldError))
            #Reset for next k/sigma Value
            allFoldRMSError = []
                    
                        
        if(curDataFrameTaskType == 'Classification'):
            print('---EDITED KNN---')
            print('K:' + str(kVal))
            for iFoldIndex in range(self.numFolds):
                print('Fold:' + str(iFoldIndex))
                loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                testDF = loopDataFrameFoldList.pop(iFoldIndex)
                trainDF = pd.concat(loopDataFrameFoldList, axis=0)

                #Run Edited KNN
                (curFoldRMSE, curFoldClassErr) = self.algoHelper.runEditedKNN(kVal, sigmaVal, epsilon, testDF, trainDF, curDataFramePredictor, curDataFrameTaskType)
                allFoldClassificationError.append(curFoldClassErr)
                    
            #Determine the overall Error for the Folds
            avgAllFoldError= sum(allFoldClassificationError)/(self.numFolds)
            print('Average Classification Error on Folds: \t' + str(avgAllFoldError))
            #Reset for next k value
            allFoldClassificationError = []      
    

    def runKFoldCrossVal_CondensedKNN_Tune(self, toRunOnDataSetName: str, kVal, sigmaVal, epsilonVals: list, zStand = False, zStandHeaders = None):
        curDataFrameFoldList = self._create_folds(self.allDataSets[toRunOnDataSetName].finalData_Validation20PercentSet)
        curDataFramePredictor = self.allDataSets[toRunOnDataSetName].predictor
        curDataFrameTaskType = self.allDataSets[toRunOnDataSetName].taskType
        
        allFoldRMSError = []
        zStandTestTrainDict = {'Train Set': None, 'Test Set': None}
        
        if(curDataFrameTaskType == 'Regression'):
            print('---Condesed KNN---')
            print('Tuning for Epsilon')
            print('K:' + str(kVal))
            print('Sigma:' + str(sigmaVal))
            for epsilon in epsilonVals:
                print('Epsilon:' + str(epsilon))
                for iFoldIndex in range(self.numFolds):
                    print('Fold:' + str(iFoldIndex))
                    loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                    testDF = loopDataFrameFoldList.pop(iFoldIndex)
                    trainDF = pd.concat(loopDataFrameFoldList, axis=0)
            
                    zStandTestTrainDict['Test Set'] = testDF
                    zStandTestTrainDict['Train Set'] = trainDF
            
                    if(zStand):
                        meanStdTuple = self.cal_mean_std(zStandTestTrainDict['Train Set'], zStandHeaders)
                        testdf_zStandTrainSet = self.zStanderdize_data(zStandTestTrainDict['Train Set'], meanStdTuple)
                        testdf_zStandTestSet = self.zStanderdize_data(zStandTestTrainDict['Test Set'], meanStdTuple)
                
                        testDF = testdf_zStandTrainSet
                        testDF = testdf_zStandTestSet
    
    
                    #INSERT RUN NORMAL KNN
                    (curFoldRMSE, curFoldClassErr) = self.algoHelper.runCondensedKNN(kVal, sigmaVal, epsilon, testDF, trainDF, curDataFramePredictor, curDataFrameTaskType)
                    allFoldRMSError.append(curFoldRMSE)
    
                #Determine the overall Error for the Folds
                avgAllFoldError= sum(allFoldRMSError)/(self.numFolds)
                print('Average Root Mean Square Error on Fold: \t' + str(avgAllFoldError))
                #Reset for next k/sigma Value
                allFoldRMSError = []
                    
    
    def runKFoldCrossVal_CondensedKNN(self, toRunOnDataSetName: str, kVal, sigmaVal, epsilon, zStand = False, zStandHeaders = None):
        curDataFrameFoldList = self._create_folds(self.allDataSets[toRunOnDataSetName].finalData_ExperimentSet)
        curDataFramePredictor = self.allDataSets[toRunOnDataSetName].predictor
        curDataFrameTaskType = self.allDataSets[toRunOnDataSetName].taskType
        
        allFoldRMSError = []
        allFoldClassificationError = []
        zStandTestTrainDict = {'Train Set': None, 'Test Set': None}
        
        if(curDataFrameTaskType == 'Regression'):
            print('---CONDENSED KNN---')
            print('K:' + str(kVal))
            print('Sigma:' + str(sigmaVal))
            print('Epsilon:' + str(epsilon))
            for iFoldIndex in range(self.numFolds):
                print('Fold:' + str(iFoldIndex))
                loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                testDF = loopDataFrameFoldList.pop(iFoldIndex)
                trainDF = pd.concat(loopDataFrameFoldList, axis=0)
            
                zStandTestTrainDict['Test Set'] = testDF
                zStandTestTrainDict['Train Set'] = trainDF
            
                if(zStand):
                    meanStdTuple = self.cal_mean_std(zStandTestTrainDict['Train Set'], zStandHeaders)
                    testdf_zStandTrainSet = self.zStanderdize_data(zStandTestTrainDict['Train Set'], meanStdTuple)
                    testdf_zStandTestSet = self.zStanderdize_data(zStandTestTrainDict['Test Set'], meanStdTuple)
                    testDF = testdf_zStandTrainSet
                    testDF = testdf_zStandTestSet

                #Run Edited KNN
                (curFoldRMSE, curFoldClassErr) = self.algoHelper.runCondensedKNN(kVal, sigmaVal, epsilon, testDF, trainDF, curDataFramePredictor, curDataFrameTaskType)
                allFoldRMSError.append(curFoldRMSE)
    
            #Determine the overall Error for the Folds
            avgAllFoldError= sum(allFoldRMSError)/(self.numFolds)
            print('Average Root Mean Square Error on Folds: \t' + str(avgAllFoldError))
            #Reset for next k/sigma Value
            allFoldRMSError = []
                    
                        
        if(curDataFrameTaskType == 'Classification'):
            print('---CONDENSED KNN---')
            print('K:' + str(kVal))
            for iFoldIndex in range(self.numFolds):
                print('Fold:' + str(iFoldIndex))
                loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                testDF = loopDataFrameFoldList.pop(iFoldIndex)
                trainDF = pd.concat(loopDataFrameFoldList, axis=0)

                #Run Edited KNN
                (curFoldRMSE, curFoldClassErr) = self.algoHelper.runCondensedKNN(kVal, sigmaVal, epsilon, testDF, trainDF, curDataFramePredictor, curDataFrameTaskType)
                allFoldClassificationError.append(curFoldClassErr)
                    
            #Determine the overall Error for the Folds
            avgAllFoldError= sum(allFoldClassificationError)/(self.numFolds)
            print('Average Classification Error on Folds: \t' + str(avgAllFoldError))
            #Reset for next k value
            allFoldClassificationError = []    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    