# Sarah Wilson 
# 303 - 921 - 7225
# Project 4
# Introduction to Machine Learning

import pandas as pd
import numpy as np
import random 

class LinearRegHelper:
    def __init__(self, dataSet):
        self.name = "Linear Regression Helper"
        self.dataSetName = dataSet.name
        self.dataSet = dataSet
        
    def runLinearRegression(self, testDF, trainDF, learningRate):
        #Runs the Linear Regression algorithm using the learning rate specificed 
        
        #Number of Columns is the _j in Figure 10.6
        numberOfColumns = len(testDF.columns)
        
        #Number of Observations is the _t in Figure 10.6
        numberOfObservations = len(trainDF)
        
        #Column Headers as a List
        columnHeaders = list(trainDF)
        
        
        #Seed the Inital Weight Values (w_j)
        for feature in range(numberOfColumns):
            weightList = []
            curWeight_J = random.uniform(-0.01, 0.01)
            weightList.append(curWeight_J)
        
        #Convert the List to a Numpy Array
        weight_J_Array = np.array(weightList, dtype=np.float)
        
        #Set the Convergence Factor
        convergenceValue = 0.1
        
        while (deltaWeight > convergenceValue):
            #Seed the inital Delta Weight values to zero (delta_w_j)
            for feature in range(numberOfColumns):
                delta_Weight_J_List = []
                curDelta_Weight_J = 0
                delta_Weight_J_List.append(curDelta_Weight_J)
                
            #Convert the List to a Numpy Array
            delta_Weight_J_Array = np.array(delta_Weight_J_List, dtype=np.float)
            
            #Start the Weight Updates
            for subT_Idx in range(numberOfObservations):
                ohValue = float(0)
                
                for subJ_Idx in range(numberOfColumns):
                    #Get the Data Frame Header 
                    featureName = columnHeaders[subJ_Idx]
                    
                    #Get the Observation 
                    observation_X_J_T = trainDF[featureName].values[subT_Idx]
                    
                    curWeight_J = delta_Weight_J_Array[subJ_Idx]
                    ohValue = ohValue + 
                
            
        
            
            
            
        

        
        