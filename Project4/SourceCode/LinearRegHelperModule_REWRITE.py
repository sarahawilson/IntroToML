# Sarah Wilson 
# 303 - 921 - 7225
# Project 4
# Introduction to Machine Learning

import pandas as pd
import numpy as np
import random 


class LinearRegHelper_REWRITE:
    def __init__(self, dataSet, numClassProblem, classA, classB):
        self.name = "Linear Regression Helper"
        self.dataSetName = dataSet.name
        self.dataSet = dataSet
        self.predictor = dataSet.predictor
        self.probType = dataSet.taskType
        self.numClassProb = numClassProblem
        self.zeroClass = classA
        self.oneClass = classB
        
    
    
    def deadSimple_LinReg(self, testDF, trainDF, N_VAL, EP_VAL):
        
        #Drop the Predictor from the trainDF
        noPred_trainDF = trainDF.drop([self.predictor], axis =1)
        
        #Number of Columns is the _j in Figure 10.6
        numberOfColumns = len(noPred_trainDF.columns)
        
        #Number of Observations is the _t in Figure 10.6
        numberOfObservations = len(noPred_trainDF)
        
        #Seed the Inital Weight Values (w_j)
        weightList = []
        for featureIdx in range(numberOfColumns):
            curWeight_J = random.uniform(-0.01, 0.01)
            weightList.append(curWeight_J)
            
        #Convert the List to a Numpy Array
        W_j_Vector = np.array(weightList, dtype=np.float)
        
        biasTerm = 0 
        
        #Let the Weights Update Until they Converge
        for curEP in range(EP_VAL):
            for observationIDx in range(numberOfObservations):
                #Build the X Vector that Represents the Observation
                curObservation = noPred_trainDF.iloc[[observationIDx]]
                X_j_Vector = curObservation.to_numpy()
                X_j_Vector = X_j_Vector[0]
                
                #Get the Actual Classifier
                actual_Y_Class = trainDF[self.predictor].values[observationIDx]
                if (self.probType == 'Regression'):
                    test = 0
                    
                elif (self.probType == 'Classification'):
                    if(actual_Y_Class == self.zeroClass):
                        actual_Y_Val = 0
                    elif (actual_Y_Class == self.oneClass):
                        actual_Y_Val = 1
                            
                
            o_val = np.dot(W_j_Vector, X_j_Vector) + biasTerm
            
            if (self.probType == 'Regression'):
                prediction_Y_Val = 0
        
            elif (self.probType == 'Classification'):
                prediction_Y_Val = 1 / (1 + np.exp(-1*o_val))
                
            
            deltaPredictionToActual = prediction_Y_Val - actual_Y_Val
            deltaWeight_GradDesc = np.dot(deltaPredictionToActual, X_j_Vector.T)
            deltaBias_GradDesc = np.sum(deltaPredictionToActual)
            W_j_Vector = W_j_Vector - (N_VAL*deltaWeight_GradDesc)
            biasTerm = biasTerm - (N_VAL*deltaBias_GradDesc)
    
        ######
        #Generate Predictions for Every Observation in the Test Set 
        
        #Use the Trained Weights on the Test Set 
        #Leverage the Test Set now
        #Drop the Predictor from the testDF
        noPred_testDF = testDF.drop([self.predictor], axis =1)
        #Number of Observations IN THE TEST SET!!!!
        numberOfObservations_Test = len(noPred_testDF)
        
        predictions_On_TestSet = []
        for observationIdx in range(numberOfObservations_Test):
            curObservation = noPred_testDF.iloc[[observationIdx]]
            X_j_Vector = curObservation.to_numpy()
            X_j_Vector = X_j_Vector[0]
            
            #Get the Actual Classifier
            actual_Y_Class = testDF[self.predictor].values[observationIdx]
            if (self.probType == 'Regression'):
                test = 0
            elif (self.probType == 'Classification'):
                if(actual_Y_Class == self.zeroClass):
                    actual_Y_Val = 0
                elif (actual_Y_Class == self.oneClass):
                    actual_Y_Val = 1
            
            
            o_val = np.dot(W_j_Vector, X_j_Vector) + biasTerm
            if (self.probType == 'Regression'):
                prediction_Y_Val = 0
        
            elif (self.probType == 'Classification'):
                prediction_Y_Val = 1 / (1 + np.exp(-1*o_val))
            
            predictions_On_TestSet.append(prediction_Y_Val)
            
        ####
        #Compare predictions on test set to actual values in the test set
        
        #Get the Actual Predictions
        actual_Test_Y_Classes = testDF[self.predictor].tolist()
        if (self.probType == 'Regression'):
            precentCorrect = 0 
            sumDiffsSqrd = 0
            for predictionIdx in range(len(predictions_On_TestSet)):
                cur_algo_pred = predictions_On_TestSet[predictionIdx]
                cur_act_class = actual_Test_Y_Classes[predictionIdx]
                
                difSqred = (cur_act_class - cur_algo_pred)**2
                sumDiffsSqrd = sumDiffsSqrd + difSqred
            
            precentCorrect = sumDiffsSqrd / (len(predictions_On_TestSet))
            
        elif (self.probType == 'Classification'):
            numCorrect = 0
            for predictionIdx in range(len(predictions_On_TestSet)):
                cur_algo_pred = predictions_On_TestSet[predictionIdx]
                cur_act_class = actual_Test_Y_Classes[predictionIdx]
                
                if(cur_algo_pred > 0.5):
                    cur_algo_pred_class = self.oneClass
                else:
                    cur_algo_pred_class = self.zeroClass
                    
                if(cur_algo_pred_class == cur_act_class):
                    numCorrect = numCorrect + 1
                    
            precentCorrect = numCorrect / (len(predictions_On_TestSet))
            
        return precentCorrect
   
        
        
 
                    
                
                    
                
                
            
        
            
            
            
        

        
        