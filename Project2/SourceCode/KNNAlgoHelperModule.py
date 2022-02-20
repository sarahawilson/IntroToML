# Sarah Wilson 
# 303 - 921 - 7225
# Project 2
# Introduction to Machine Learning

from typing import List, Tuple, Dict
from numpy import array, argmin, sqrt, sum
import numpy as np

class KNNAlgoHelper:
    def __init__(self):
        self.name = 'KNN Helper'

    def RunNormalKNN(self, 
                          kval: int,
                          sigmaVal,
                          testSet, 
                          trainSet, 
                          predictor: str, 
                          taskType: str):
        k = kval;
        sigma = sigmaVal
        unmodTestSet = testSet
        unmodTrainSet = trainSet
        # Drop the Predictor from the data frame since we don't want 
        # it included in the distance calculations
        testSet = testSet.drop(columns=predictor)
        trainSet = trainSet.drop(columns=predictor)
        
        testSetArray = testSet.to_numpy()
        trainSetArray = trainSet.to_numpy()
        
        #Get the number of rows in the testSetArray 
        numRowsTestSet = testSetArray.shape[0]
        
        classificationWrongCnt = 0
        regressionSumErrorsSqrd = 0
        
        for curRow in range(numRowsTestSet):
            testSetCurRow = testSetArray[curRow]
            
            #loopDataFrameFoldList = copy.deepcopy(curDataFrameFoldList)
            #testDF = loopDataFrameFoldList.pop(iFoldIndex)
            #trainDF = pd.concat(loopDataFrameFoldList, axis=0)
            
            
            curRowDiff = testSetCurRow - trainSetArray
            curRowDist = sqrt(sum(curRowDiff**2,axis=-1))
            #dist - rows correspond to the datapoint in the Train Set
            #dist - colms correspond to the datapoint in the Test Set
            
            #Get the Index of the smallest values  
            minDistanceIndexAll = np.argpartition(curRowDist, k)
            minDistanceKNeighborsIndex = minDistanceIndexAll[:k]
            
            #Get the Predictor of each of the K Nearest Neighbors
            #in the Train Set
            kNNPredictors = []
            for nearNeighborIdx in minDistanceKNeighborsIndex:
                kNNPredictors.append(unmodTrainSet.iloc[nearNeighborIdx][predictor])
                
            #Get the Query Point (Test Set) predictor
            curQueryPredictor = unmodTestSet.iloc[curRow][predictor]
            
            #Compare the Query Predictor to the KNN Predictors
            if(taskType == 'Regression'):
                minDistanceKNeighborsValues = curRowDist[minDistanceKNeighborsIndex[:k]]
                #Apply the Gaussian Kernel
                powerOf = (-1/(2*sigma))*minDistanceKNeighborsValues
                gKernelMinDistanceKNNValues = np.exp(powerOf)
                
                wAvgNumerator = np.float64(0)
                wAvgDenomnator = sum(gKernelMinDistanceKNNValues)
                for index in range(len(kNNPredictors)):
                    wAvgNumerator = wAvgNumerator + (kNNPredictors[index] * gKernelMinDistanceKNNValues[index])
                
                if(wAvgDenomnator != 0):
                    weightedAvg = wAvgNumerator / wAvgDenomnator
                else:
                    weightedAvg = 1                
                
                curRegErr = (curQueryPredictor - weightedAvg)**2
                regressionSumErrorsSqrd = regressionSumErrorsSqrd + curRegErr
                
            #For Classification Take the Most Common out of the 
            #nearest neighbors
            elif(taskType == 'Classification'):
                mostCommon = max(kNNPredictors, key = kNNPredictors.count)
                #print('\t' + curQueryPredictor)
                #print('\t Most Common:' + mostCommon)
                if(mostCommon != curQueryPredictor):
                    classificationWrongCnt = classificationWrongCnt + 1
            
        
        if(taskType == 'Regression'):
            regressionError = None
            regressionMSE = sqrt(regressionSumErrorsSqrd / numRowsTestSet)
            print('\t' + 'RMSE: ' + str(regressionMSE))
            return (regressionMSE, regressionError)
        
        elif(taskType == 'Classification'):
            classificationError = classificationWrongCnt / numRowsTestSet
            classificationMSE = None #Not applied for classificaiton taks
            print('\t' + 'Classification Error: ' + str(classificationError))
            return (classificationMSE, classificationError)


    def runEditedKNN(self, 
                     kVal: int,
                     sigmaVal: int,
                     epsilonVal: int
                     testSet, 
                     trainSet, 
                     predictor: str,
                     taskType: str):
        
        k = kVal;
        sigma = sigmaVal
        unmodTestSet = testSet
        unmodTrainSet = trainSet
            
        # Drop the Predictor from the data frame since we don't want 
        # it included in the distance calculations
        testSet = testSet.drop(columns=predictor)
        trainSet = trainSet.drop(columns=predictor)
            
        #Convert to Numpy Array
        trainSetArray = trainSet.to_numpy()
            
        #Get the number of rows in the testSetArray 
        numRowsTrainSet = trainSetArray.shape[0]
        
        #TrainSet Indexs to Drop
        trainSetIdxToDrop = []
            
        for curRow in range(numRowsTrainSet):
            trainSetCurRow = trainSetArray[curRow]
            
            #Delete the current train point out of the train set 
            #so it is not accounted for in distance cals,
            modTrainSetArray = np.delete(trainSetArray, curRow, 0)
            
            curRowDiff = trainSetCurRow - modTrainSetArray
            curRowDist = sqrt(sum(curRowDiff**2,axis=-1))
            #dist - rows correspond to the datapoint in the Train Set
            #dist - colms correspond to the datapoint in the Test Set
            
            #Get the Index of the smallest values  
            minDistanceIndexAll = np.argpartition(curRowDist, k)
            minDistanceKNeighborsIndex = minDistanceIndexAll[:k]
            
            #Get the Predictor of each of the K Nearest Neighbors
            #in the Train Set
            kNNPredictors = []
            for nearNeighborIdx in minDistanceKNeighborsIndex:
                kNNPredictors.append(unmodTrainSet.iloc[nearNeighborIdx][predictor])
                
            #Get the Query Point (Train Set) predictor
            curQueryPredictor = unmodTrainSet.iloc[curRow][predictor]
            
            #Compare the Query Predictor to the KNN Predictors
            if(taskType == 'Regression'):
                minDistanceKNeighborsValues = curRowDist[minDistanceKNeighborsIndex[:k]]
                #Apply the Gaussian Kernel
                powerOf = (-1/(2*sigma))*minDistanceKNeighborsValues
                gKernelMinDistanceKNNValues = np.exp(powerOf)
                
                wAvgNumerator = np.float64(0)
                wAvgDenomnator = sum(gKernelMinDistanceKNNValues)
                for index in range(len(kNNPredictors)):
                    wAvgNumerator = wAvgNumerator + (kNNPredictors[index] * gKernelMinDistanceKNNValues[index])
                
                if(wAvgDenomnator != 0):
                    weightedAvg = wAvgNumerator / wAvgDenomnator
                else:
                    weightedAvg = 1  
                    
                #Compare to the Epsilon Value
                if(np.abs(curQueryPredictor - weightedAvg) > epsilonVal):
                    #If greater than Epsilon we want to drop this current query point from the data set
                    trainSetIdxToDrop.append(curRow)
                    
                
                
            #For Classification Take the Most Common out of the 
            #nearest neighbors
            elif(taskType == 'Classification'):
                mostCommon = max(kNNPredictors, key = kNNPredictors.count)
                #print('\t' + curQueryPredictor)
                #print('\t Most Common:' + mostCommon)
                if(mostCommon != curQueryPredictor):
                    #If not equal to the most common then we want to drop this current query point from the data set
                    trainSetIdxToDrop.append(curRow)
                    
        #Drop the Index from the Train Set
        editedTrainSet = np.delete(trainSetArray, trainSetIdxToDrop, 0)
        
        
        #Run KNN on thisnew Edited Train Set
        
        #Get the number of rows in the testSetArray 
        numRowsTestSet = testSetArray.shape[0]
        
        classificationWrongCnt = 0
        regressionSumErrorsSqrd = 0
        
        for curRow in range(numRowsTestSet):
            testSetCurRow = testSetArray[curRow]
            
            curRowDiff = testSetCurRow - editedTrainSet
            curRowDist = sqrt(sum(curRowDiff**2,axis=-1))
            #dist - rows correspond to the datapoint in the Train Set
            #dist - colms correspond to the datapoint in the Test Set
            
            #Get the Index of the smallest values  
            minDistanceIndexAll = np.argpartition(curRowDist, k)
            minDistanceKNeighborsIndex = minDistanceIndexAll[:k]
            
            #Get the Predictor of each of the K Nearest Neighbors
            #in the Train Set
            kNNPredictors = []
            for nearNeighborIdx in minDistanceKNeighborsIndex:
                kNNPredictors.append(unmodTrainSet.iloc[nearNeighborIdx][predictor])
                
            #Get the Query Point (Test Set) predictor
            curQueryPredictor = unmodTestSet.iloc[curRow][predictor]
            
            #Compare the Query Predictor to the KNN Predictors
            if(taskType == 'Regression'):
                minDistanceKNeighborsValues = curRowDist[minDistanceKNeighborsIndex[:k]]
                #Apply the Gaussian Kernel
                powerOf = (-1/(2*sigma))*minDistanceKNeighborsValues
                gKernelMinDistanceKNNValues = np.exp(powerOf)
                
                wAvgNumerator = np.float64(0)
                wAvgDenomnator = sum(gKernelMinDistanceKNNValues)
                for index in range(len(kNNPredictors)):
                    wAvgNumerator = wAvgNumerator + (kNNPredictors[index] * gKernelMinDistanceKNNValues[index])
                
                if(wAvgDenomnator != 0):
                    weightedAvg = wAvgNumerator / wAvgDenomnator
                else:
                    weightedAvg = 1                
                
                curRegErr = (curQueryPredictor - weightedAvg)**2
                regressionSumErrorsSqrd = regressionSumErrorsSqrd + curRegErr
                
            #For Classification Take the Most Common out of the 
            #nearest neighbors
            elif(taskType == 'Classification'):
                mostCommon = max(kNNPredictors, key = kNNPredictors.count)
                #print('\t' + curQueryPredictor)
                #print('\t Most Common:' + mostCommon)
                if(mostCommon != curQueryPredictor):
                    classificationWrongCnt = classificationWrongCnt + 1
            
        
        if(taskType == 'Regression'):
            regressionError = None
            regressionMSE = sqrt(regressionSumErrorsSqrd / numRowsTestSet)
            print('\t' + 'RMSE: ' + str(regressionMSE))
            return (regressionMSE, regressionError)
        
        elif(taskType == 'Classification'):
            classificationError = classificationWrongCnt / numRowsTestSet
            classificationMSE = None #Not applied for classificaiton taks
            print('\t' + 'Classification Error: ' + str(classificationError))
            return (classificationMSE, classificationError)
            
                    




            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            