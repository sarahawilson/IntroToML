# Sarah Wilson 
# 303 - 921 - 7225
# Project 4
# Introduction to Machine Learning

import pandas as pd
import numpy as np
import random 
import math

class LinearRegHelper:
    def __init__(self, dataSet):
        self.name = "Linear Regression Helper"
        self.dataSetName = dataSet.name
        self.dataSet = dataSet
        self.predictor = dataSet.predictor
        
    def runLinearRegression(self, testDF, trainDF, learningRate):
        #Runs the Linear Regression algorithm using the learning rate specificed 
        
        #Number of Columns is the _j in Figure 10.6
        numberOfColumns = len(trainDF.columns)
        
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
        
        #Save off a Copy of the Orignal Weight Vecotr
        orginial_weight_J_Array = weight_J_Array
        
        #Set the Convergence Factor
        convergenceValue = 0.1
        
        while (deltaWeight > convergenceValue):
            #Seed the inital Delta Weight values to zero (delta_w_j)
            delta_Weight_J_List = []
            for feature in range(numberOfColumns):
                curDelta_Weight_J = 0
                delta_Weight_J_List.append(curDelta_Weight_J)
                
            #Convert the List to a Numpy Array
            delta_Weight_J_Array = np.array(delta_Weight_J_List, dtype=np.float)
            

            #Loop over the number of observations
            for subT_Idx in range(numberOfObservations):
                oddsValue = float(0)
                
                #Loop over the number of features (dims)
                for subJ_Idx in range(numberOfColumns):
                    #Get the Data Frame Header 
                    featureName = columnHeaders[subJ_Idx]
                    
                    #Get the Observation value for that feature
                    observation_X_J_T = trainDF[featureName].values[subT_Idx]
                    
                    curWeight_J = weight_J_Array[subJ_Idx]
                    oddsValue = oddsValue + (observation_X_J_T * curWeight_J)
                
                #Apply the Sigmoid Funciton
                curY = 1 / (1 + (math.exp(-1*oddsValue)))
                #yPrediction.append(curY)
                
                #Loop over the number of features (dims)
                for subJ_Idx in range(numberOfColumns):
                    curDelta_Weight =  delta_Weight_J_Array[subJ_Idx]
                    curObservationClass = trainDF[self.predictor].values[subT_Idx]
                    curObservation_X_J_T = trainDF[featureName].values[subT_Idx]
                    
                    updatedDelta_Weight = curDelta_Weight + ((curObservationClass - curY)*curObservation_X_J_T)
                    
                    #Update the Delta Weight Array with the new value
                    delta_Weight_J_Array[subJ_Idx] = updatedDelta_Weight
             
            #Loop over the number of features (dims)
            for subJ_Idx in range(numberOfColumns):
                curWeight_J = weight_J_Array[subJ_Idx]
                
                updatedWeight_J = curWeight_J + (learningRate * curWeight_J)
                
                #Update the Weight Array with the new value
                weight_J_Array[subJ_Idx] = updatedWeight_J
                    
                
                    
                
                
            
        
            
            
            
        

        
        