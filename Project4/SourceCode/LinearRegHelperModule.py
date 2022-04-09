# Sarah Wilson 
# 303 - 921 - 7225
# Project 4
# Introduction to Machine Learning

import pandas as pd
import numpy as np
import random 
import math

class LinearRegHelper:
    def __init__(self, dataSet, numClassProblem, classA, classB):
        self.name = "Linear Regression Helper"
        self.dataSetName = dataSet.name
        self.dataSet = dataSet
        self.predictor = dataSet.predictor
        self.probType = dataSet.taskType
        self.numClassProb = numClassProblem
        self.zeroToPntFive = classA
        self.abovePntFive = classB
        
    
    def reportError_LinearReg(self, testDF, trainDF, N_VAL, EP_VAL):
        linReg_Test_Set_Predicitions = self.runLinearRegression(testDF, trainDF, N_VAL, EP_VAL)
        #print(linReg_Test_Set_Predicitions)
        
        
        #Get the Actual Predictions
        actual_Test_Set_Values = testDF[self.predictor].tolist()
        #print(actual_Test_Set_Values)
        
        if (self.probType == 'Regression'):
            precentCorrect = 0 
            sumDiffsSqrd = 0
            for predictionIdx in range(len(linReg_Test_Set_Predicitions)):
                cur_algo_pred = linReg_Test_Set_Predicitions[predictionIdx]
                cur_act_class = actual_Test_Set_Values[predictionIdx]
                
                difSqred = (cur_act_class - cur_algo_pred)**2
                sumDiffsSqrd = sumDiffsSqrd + difSqred
            
            precentCorrect = np.sqrt(sumDiffsSqrd / (len(linReg_Test_Set_Predicitions)))
            
        elif (self.probType == 'Classification'):
            numCorrect = 0
            for predictionIdx in range(len(linReg_Test_Set_Predicitions)):
                cur_algo_pred = linReg_Test_Set_Predicitions[predictionIdx]
                cur_act_class = actual_Test_Set_Values[predictionIdx]
                
                if(cur_algo_pred > 0.5):
                    cur_algo_pred_class = self.abovePntFive
                else:
                    cur_algo_pred_class = self.zeroToPntFive
                    
                if(cur_algo_pred_class == cur_act_class):
                    numCorrect = numCorrect + 1
                    
            precentCorrect = numCorrect / (len(linReg_Test_Set_Predicitions))
            
        return precentCorrect
                    
                    
    
    
    def runLinearRegression(self, testDF, trainDF, N_VAL, EP_VAL):
        #Run Linear Regression
        # Uses the trainDF to first deteremine the Weights needed for the model
        # weights are determined by using the Learning Rate (N_VAL)
        # and the Convergence Factor (EP_VAL)
        # Then uses the testDF to predict the classifer or predictor on (basically the output of the model)
        predictions_On_TestSet = []

        #Determine the Weights
        Weights_J_Vector = self.determine_weights(trainDF, N_VAL, EP_VAL)
        #print(Weights_J_Vector)
        
        #Leverage the Test Set now
        #Drop the Predictor from the testDF
        noPred_testDF = testDF.drop([self.predictor], axis =1)
        #Number of Observations IN THE TEST SET!!!!
        numberOfObservations_Test = len(noPred_testDF)
        
        for observationIdx in range(numberOfObservations_Test):
            curObservation = noPred_testDF.iloc[[observationIdx]]
            X_j_Vector = curObservation.to_numpy()
            X_j_Vector = X_j_Vector[0]
            
            prediction = self._calcRegPrediction(X_j_Vector, Weights_J_Vector)
            predictions_On_TestSet.append(prediction)
        return predictions_On_TestSet
            
       
    def determine_weights(self, trainDF, N_VAL, EP_VAL):
        #Drop the Predictor from the trainDF
        noPred_trainDF = trainDF.drop([self.predictor], axis =1)
        
        #Number of Columns is the _j in Figure 10.6
        numberOfColumns = len(noPred_trainDF.columns)
        
        #Number of Observations is the _t in Figure 10.6
        numberOfObservations = len(noPred_trainDF)
        
        
        #Seed the Inital Weight Values (w_j)
        weightList = [1] #This is the Bais, assume the Bais is always one
        for featureIdx in range(numberOfColumns):
            curWeight_J = random.uniform(-0.01, 0.01)
            weightList.append(curWeight_J)
            
        #Convert the List to a Numpy Array
        W_j_Vector = np.array(weightList, dtype=np.float)
        
        for curEP in range(EP_VAL):
            for observationIDx in range(numberOfObservations):
                curObservation = noPred_trainDF.iloc[[observationIDx]]
                X_j_Vector = curObservation.to_numpy()
                X_j_Vector = X_j_Vector[0]
                
                #Get the Weighted Sum of this Row (O_VAL)
                O_Val = self._calcWeightSumRow_OVAL(numberOfColumns, W_j_Vector, X_j_Vector)
                
                if(self.probType == 'Classification'):
                    #Take the Sigmod of O_VAL to get the Predicted Value Y_VAL
                    Y_Val = 1 / (1 + math.exp(-1*O_Val))
                
                    #Convert the Y_Val to be a Class Label
                    if(Y_Val > 0.5):
                        Y_Val_Predictor = self.abovePntFive
                    else:
                        Y_Val_Predictor = self.zeroToPntFive
                        
                elif(self.probType == 'Regression'):
                    Y_Val_Predictor = O_Val
                    
                    
                #Get the Current Observation Predictor
                curObservationPredictor = trainDF[self.predictor].values[observationIDx]
                
                #Calculate the Error between predicted and actual
                if(self.probType == 'Classification'):
                    if(curObservationPredictor == Y_Val_Predictor):
                        error = 0
                    elif(curObservationPredictor != Y_Val_Predictor):
                        error = 1 
                        
                elif(self.probType == 'Regression'):
                    error = (curObservationPredictor - Y_Val_Predictor)
               
                #Update the Bias
                W_j_Vector[0] = W_j_Vector[0] + (N_VAL * error)
                #Set the Bias always to one maybe??
                #W_j_Vector[0] = 1
               
                #Update the Other Weights based on the error
                for colIdx in range(numberOfColumns):
                    W_j_Vector[colIdx + 1] = W_j_Vector[colIdx + 1] + (N_VAL * error * X_j_Vector[colIdx])
        
        return W_j_Vector      
        
        
    def _calcWeightSumRow_OVAL(self, numberOfColumns, W_j_Vector, X_j_Vector):
        O_VAL = 0;
        for colIdx in range(numberOfColumns):
            O_VAL = O_VAL + (W_j_Vector[colIdx+1] * X_j_Vector[colIdx])
        return O_VAL 
    
    def _calcRegPrediction(self, X_j_Vector, W_j_Vector):
        curPrediction = W_j_Vector[0] 
        #curPrediction = 0 #When bias is always set to one 
        for colIdx in range(len(X_j_Vector)):
            curPrediction = curPrediction + (W_j_Vector[colIdx +1] * X_j_Vector[colIdx])
        
        if(self.probType == 'Classification'):
            #Apply the Sigmoid Function
            sigmoidPrediction = 1 / (1 + math.exp(-1*curPrediction))
            madePrediction = sigmoidPrediction
            #print(sigmoidPrediction)
        elif(self.probType == 'Regression'):
            madePrediction = curPrediction
        
        
        return madePrediction
   
        
        
 
                    
                
                    
                
                
            
        
            
            
            
        

        
        