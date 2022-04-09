# Sarah Wilson 
# 303 - 921 - 7225
# Project 4
# Introduction to Machine Learning

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import copy
import LinearRegHelperModule
import LinearRegHelperModule_REWRITE
import LinearNNHelperModule


class KCrossValHelper:
    def __init__(self,
                 allDataSets: Dict,
                 ):
        
        self.name = 'KCrossVal'
        self.numFolds = 5
        self.allDataSets = allDataSets
        self._createValidation_TuneAndExperimentSets()
                
        
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
    
    def DEADSIMPLE_runKFoldCrossVal_Linear_Regression_Tune(self, dataSetName: str, nVals: list, epVals: list, numClassProblem, classA, classB):
        #Runs the Linear Regression algorithm using 5 fold cross validation
        
        linRegHelper = LinearRegHelperModule_REWRITE.LinearRegHelper_REWRITE(self.allDataSets[dataSetName],numClassProblem, classA, classB)
        
        curDataFrameFoldList = self._create_folds(self.allDataSets[dataSetName].finalData_Validation20PercentSet)
        
        print('---TUNE LINEAR REGRESSION---')
        print('Tuning Learning Factor (N) On: ' + dataSetName)
        print('Tuning Convergence Factor (EP) On: ' + dataSetName)
        for nVal in nVals:
            print('N:' + str(nVal))
            for epVal in epVals:
                print('EP:' + str(epVal))
                curFoldResultList = []
                for iFoldIndex in range(self.numFolds):
                    print('Fold:' + str(iFoldIndex))
                    loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                    testDF = loopDataFrameFoldList.pop(iFoldIndex)
                    trainDF = pd.concat(loopDataFrameFoldList, axis=0)
                    error = linRegHelper.deadSimple_LinReg(testDF, trainDF, nVal, epVal)
                    curFoldResultList.append(error)
                    print('\t Fold Results:' + str(error))
                
                #Calcaulte the Average Across the Folds
                curSum = 0
                for res in curFoldResultList:
                    curSum = curSum + res
                resultAvg = curSum / self.numFolds
                print('Folds Average:' + str(resultAvg))

    
    def runKFoldCrossVal_Linear_Regression_Tune(self, dataSetName: str, nVals: list, epVals: list, numClassProblem, classA, classB):
        #Runs the Linear Regression algorithm using 5 fold cross validation
        
        linRegHelper = LinearRegHelperModule.LinearRegHelper(self.allDataSets[dataSetName],numClassProblem, classA, classB)
        
        curDataFrameFoldList = self._create_folds(self.allDataSets[dataSetName].finalData_Validation20PercentSet)
        
        print('---TUNE LINEAR REGRESSION---')
        print('Tuning Learning Factor (N) On: ' + dataSetName)
        print('Tuning Convergence Factor (EP) On: ' + dataSetName)
        for nVal in nVals:
            print('N:' + str(nVal))
            for epVal in epVals:
                print('EP:' + str(epVal))
                curFoldResultList = []
                for iFoldIndex in range(self.numFolds):
                    print('Fold:' + str(iFoldIndex))
                    loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
                    testDF = loopDataFrameFoldList.pop(iFoldIndex)
                    trainDF = pd.concat(loopDataFrameFoldList, axis=0)
                    error = linRegHelper.reportError_LinearReg(testDF, trainDF, nVal, epVal)
                    curFoldResultList.append(error)
                    print('\t Fold Results:' + str(error))
                
                #Calcaulte the Average Across the Folds
                curSum = 0
                for res in curFoldResultList:
                    curSum = curSum + res
                resultAvg = curSum / self.numFolds
                print('Folds Average:' + str(resultAvg))
                
            

            
    def runKFoldCrossVal_Linear_Regression(self, dataSetName: str, nVal, epVal, numClassProblem, classA, classB):
        #Runs the Linear Regression algorithm using 5 fold cross validation after the 
        #Tuning process has occured
        
        linRegHelper = LinearRegHelperModule_REWRITE.LinearRegHelper_REWRITE(self.allDataSets[dataSetName],numClassProblem, classA, classB)
        
        curDataFrameFoldList = self._create_folds(self.allDataSets[dataSetName].finalData_ExperimentSet)
        
        print('---LINEAR REGRESSION---')
        print('DataSet: ' + dataSetName)
        print('Learning Factor (N): ' + str(nVal))
        print('Convergence Factor (EP): ' + str(epVal))
        curFoldResultList = []
        for iFoldIndex in range(self.numFolds):
            print('Fold:' + str(iFoldIndex))
            loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
            testDF = loopDataFrameFoldList.pop(iFoldIndex)
            trainDF = pd.concat(loopDataFrameFoldList, axis=0)   
            error = linRegHelper.deadSimple_LinReg(testDF, trainDF, nVal, epVal)
            curFoldResultList.append(error)
            print('\t Fold Results:' + str(error))
                
        #Calcaulte the Average Across the Folds
        curSum = 0
        for res in curFoldResultList:
            curSum = curSum + res
        resultAvg = curSum / self.numFolds
        print('Folds Average:' + str(resultAvg))
        
        
    def runKFoldCrossVal_Linear_NN(self, dataSetName: str, numClassProblem, classA, classB):
        #Runs the Linear Regression algorithm using 5 fold cross validation after the 
        #Tuning process has occured
        
        nnHelper = LinearNNHelperModule.Linear_NN_Helper(self.allDataSets[dataSetName], numClassProblem, classA, classB)
        
        
        #Get the Size of the Input (minus one to account for the predictor column)
        numberOfInputs = len(self.allDataSets[dataSetName].finalData_ExperimentSet.columns) - 1
        
        outputNodes = 1

        network = nnHelper.build_template_network(numberOfInputs, 2, numberOfInputs, outputNodes)
        
        
        #Quick Test - On the feed foward
        curObservation = self.allDataSets[dataSetName].finalData_ExperimentSet.iloc[[0]]
        curObservation_noPred = curObservation.drop(['Class'], axis =1)
        row = curObservation_noPred.to_numpy()
        row = row[0]
        test =1 
        
        output = nnHelper.feedforward_prop(row, network)
        
        
        #Quick Test - On the Back Prop
        actual_Y_Class = self.allDataSets[dataSetName].finalData_ExperimentSet['Class'].values[0]
        nnHelper.backwards_prop(actual_Y_Class)
        
        
        
        
        
        
        
        
        #curDataFrameFoldList = self._create_folds(self.allDataSets[dataSetName].finalData_ExperimentSet)
        
#        print('---LINEAR NN---')
#        print('DataSet: ' + dataSetName)
#        print('Learning Factor (N): ' + str(nVal))
#        print('Convergence Factor (EP): ' + str(epVal))
#        curFoldResultList = []
#        for iFoldIndex in range(self.numFolds):
#            print('Fold:' + str(iFoldIndex))
#            loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
#            testDF = loopDataFrameFoldList.pop(iFoldIndex)
#            trainDF = pd.concat(loopDataFrameFoldList, axis=0)   
#            error = linRegHelper.deadSimple_LinReg(testDF, trainDF, nVal, epVal)
#            curFoldResultList.append(error)
#            print('\t Fold Results:' + str(error))
#                
#        #Calcaulte the Average Across the Folds
#        curSum = 0
#        for res in curFoldResultList:
#            curSum = curSum + res
#        resultAvg = curSum / self.numFolds
#        print('Folds Average:' + str(resultAvg))
            


        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    