# Sarah Wilson 
# 303 - 921 - 7225
# Project 2
# Introduction to Machine Learning

from typing import List, Tuple, Dict
import pandas as pd
from numpy import array, argmin, sqrt, sum
import numpy as np
from collections import Counter

class KNNAlgoHelper:
    def __init__(self,
                 allDataSets: Dict):
        
        self.name = 'KNN'
        self.allDataSets = allDataSets
        self._kNeighbors = None # Can be updated after tuning 
        
        #Convert All Data Sets to Numeric
        self._convertAllDataSetsToNumeric()
        
        
    
    def _convertAllDataSetsToNumeric(self):
        #Take the Nominal Data in the Data Sets and Applies One Hot Encoding
        #Takes the Ordinal Data in the Data Sets and Applies One Hot Encoding
        
        #Define the Tuple for all the data that needs to be one hot encoded
        toApplyOneHotOn =[('Albalone', ['Sex']), 
                        ('Computer Hardware', ['Vendor Name', 'Model Name']),
                        ('Forest Fire', ['month', 'day'])
                        ]
        
        
        
        carEvalOrdinalEncoding = {'Buying': {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1},
                                  'Maint': {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1},
                                  'Lug_Boot': {'big': 3, 'med': 2, 'small': 1},
                                  'Safety': {'high': 3, 'med': 2, 'low': 1}
                                  }
        
        
        congVoteOrdinalEncoding = {'handicapped-infants': {'y': 1, '?':0, 'n':-1}, 
                           'water-project-cost-sharing': {'y': 1, '?':0, 'n':-1},  
                           'adoption-of-the-budget-resolution': {'y': 1, '?':0, 'n':-1},  
                           'physician-fee-freeze': {'y': 1, '?':0, 'n':-1}, 
                           'el-salvador-aid': {'y': 1, '?':0, 'n':-1}, 
                           'religious-groups-in-schools': {'y': 1, '?':0, 'n':-1}, 
                           'anti-satellite-test-ban': {'y': 1, '?':0, 'n':-1}, 
                           'aid-to-nicaraguan-contras': {'y': 1, '?':0, 'n':-1}, 
                           'mx-missile': {'y': 1, '?':0, 'n':-1},
                           'immigration': {'y': 1, '?':0, 'n':-1}, 
                           'synfuels-corporation-cutback': {'y': 1, '?':0, 'n':-1}, 
                           'education-spending': {'y': 1, '?':0, 'n':-1}, 
                           'superfund-right-to-sue': {'y': 1, '?':0, 'n':-1}, 
                           'crime':{'y': 1, '?':0, 'n':-1},  
                           'duty-free-exports':{'y': 1, '?':0, 'n':-1},  
                           'export-administration-act-south-africa':{'y': 1, '?':0, 'n':-1}}
        

        #Define the Tuple for all the data sets that need to be Ordinal Encoded
        toApplyOrdinalEncodingOn = [('Car Eval',carEvalOrdinalEncoding),
                                    ('Congressional Vote', congVoteOrdinalEncoding)
                                    ]
        
        #Congressional Vote
        #TODO: NEED TO FIGURE OUT WHAT TO DO WITH CONG VOTE DATA
        # Skipping for now need to figure out 
        
        # Loop over the data sets and apply the 
        # One hot encoding to those that need it 
        # Based on the toApplyOneHotOn Tuple above
        for dataSetName in self.allDataSets:
            for curTuple in toApplyOneHotOn:
                applyOnDataSetName = curTuple[0]
                if (applyOnDataSetName == dataSetName):
                    self.allDataSets[dataSetName].applyOneHotEncoding(curTuple[1])
                    break

            #Now apply the Ordinal Data Encoding
            for curOrdTuple in toApplyOrdinalEncodingOn:
                applyOrdOnDataSetName = curOrdTuple[0]
                if (applyOrdOnDataSetName == dataSetName):
                    self.allDataSets[dataSetName].finalData.replace(to_replace=curOrdTuple[1], inplace = True)
      
    def setKNeighborsParam(self, inputKNeighbors):
        self._kNeighbors = inputKNeighbors
        
    def getKNeighborsParam(self):
        return self._kNeighbors
                      

    def runKNN_Algorithm(self, 
                          kval: int, 
                          testSet, 
                          trainSet, 
                          predictor: str, 
                          taskType: str):
        k = kval;
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
                avgClosest = sum(kNNPredictors)/k
                curRegErr = (curQueryPredictor - avgClosest)**2
                regressionSumErrorsSqrd = regressionSumErrorsSqrd + curRegErr
            elif(taskType == 'Classification'):
                mostCommon = max(kNNPredictors, key = kNNPredictors.count)
                #print('\t' + curQueryPredictor)
                #print('\t Most Common:' + mostCommon)
                if(mostCommon != curQueryPredictor):
                    classificationWrongCnt = classificationWrongCnt + 1
            
        
        if(taskType == 'Regression'):
            regressionError = None
            regressionMSE = regressionSumErrorsSqrd / numRowsTestSet
            print('\t' + 'Regression MSE: ' + str(regressionMSE))
            return (regressionMSE, regressionError)
        elif(taskType == 'Classification'):
            classificationError = classificationWrongCnt / numRowsTestSet
            classificationMSE = None #Not applied for classificaiton taks
            print('\t' + 'Classification Error: ' + str(classificationError))
            return (classificationMSE, classificationError)
        
    def testRunKNN(self):
        
        testSetArray = array([1,1,1])
        #rowsTestSetArray, colsTestSetArray = testSetArray.shape
        
        trainSetArray = array([[0,0,1], [1,0,0], [0,0,0], [0,0,0]])
        #rowTrainSetArray, colsTrainSetArray = trainSetArray.shape
        
        rowDiff = testSetArray - trainSetArray
        print(rowDiff)
        print(rowDiff**2)
        print(sum(rowDiff**2))
        print(sqrt(sum(rowDiff**2)))
        dist = sqrt(sum(rowDiff**2,axis=-1))
        print(dist)
        

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            