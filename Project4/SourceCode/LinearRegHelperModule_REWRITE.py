# Sarah Wilson 
# 303 - 921 - 7225
# Project 4
# Introduction to Machine Learning

import pandas as pd
import numpy as np
import random 
import math


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
        self.twoClass = 'good'
        self.threeClass = 'vgood'
        
    
    
    def deadSimple_LinReg(self, testDF, trainDF, N_VAL, EP_VAL):
        #Uses the Numpy package to do very simple
        #linear regresion calculations
        
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

        #Let the Weights Update Until they Converge
        for curEP in range(EP_VAL):
            
            #Seed the Inital Delta Weight Values (delta_w_j)
            deltaWeightList = []
            for featureIdx in range(numberOfColumns):
                curDelta_Weight_J = 0
                deltaWeightList.append(curDelta_Weight_J)
            
            #Convert the List to a Numpy Array
            Delta_W_j_Vector = np.array(deltaWeightList, dtype=np.float)
            biasTerm = 0
            delta_bais = 0
            
            for observationIDx in range(numberOfObservations):
                #Build the X Vector that Represents the Observation
                curObservation = noPred_trainDF.iloc[[observationIDx]]
                X_j_Vector = curObservation.to_numpy()
                X_j_Vector = X_j_Vector[0]
                
                #Get the Actual Classifier
                actual_Y_Class = trainDF[self.predictor].values[observationIDx]
                if (self.probType == 'Regression'):
                    actual_Y_Val = actual_Y_Class
                    
                elif (self.probType == 'Classification'):
                    if(actual_Y_Class == self.zeroClass):
                        actual_Y_Val = 0
                    elif (actual_Y_Class == self.oneClass):
                        actual_Y_Val = 1
                            
                
                o_val = np.dot(W_j_Vector, X_j_Vector) + biasTerm
            
                if (self.probType == 'Regression'):
                    prediction_Y_Val = o_val
        
                elif (self.probType == 'Classification'):
                    prediction_Y_Val = 1 / (1 + np.exp(-1*o_val))
                
            
                deltaPredictionToActual = prediction_Y_Val - actual_Y_Val
                
                deltaWeight_GradDesc = np.dot(deltaPredictionToActual, X_j_Vector.T)
                
                Delta_W_j_Vector = Delta_W_j_Vector + deltaWeight_GradDesc
                
                delta_bais = 1 #delta_bais + deltaPredictionToActual
                
            #deltaBias_GradDesc = np.sum(deltaPredictionToActual)
            #W_j_Vector = W_j_Vector + (N_VAL*deltaWeight_GradDesc)
            #print('PreUpdate')
            #print(W_j_Vector)
            W_j_Vector = W_j_Vector + (N_VAL*Delta_W_j_Vector)
            #print('PreUpdate')
            #print(W_j_Vector)
            #biasTerm = biasTerm - (N_VAL*deltaBias_GradDesc)
            biasTerm = biasTerm + (N_VAL * delta_bais)
    
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
                actual_Y_Val = actual_Y_Class
            elif (self.probType == 'Classification'):
                if(actual_Y_Class == self.zeroClass):
                    actual_Y_Val = 0
                elif (actual_Y_Class == self.oneClass):
                    actual_Y_Val = 1
            
            
            o_val = np.dot(W_j_Vector, X_j_Vector) + biasTerm
            if (self.probType == 'Regression'):
                prediction_Y_Val = o_val
        
            elif (self.probType == 'Classification'):
                prediction_Y_Val = 1 / (1 + np.exp(-1*o_val))
            
            predictions_On_TestSet.append(prediction_Y_Val)
            
        ####
        #Compare predictions on test set to actual values in the test set
        
        #print(predictions_On_TestSet)
        #Get the Actual Predictions
        actual_Test_Y_Classes = testDF[self.predictor].tolist()
        #print(actual_Test_Y_Classes)
        if (self.probType == 'Regression'):
            precentCorrect = 0 
            sumDiffsSqrd = 0
            for predictionIdx in range(len(predictions_On_TestSet)):
                cur_algo_pred = predictions_On_TestSet[predictionIdx]
                cur_act_class = actual_Test_Y_Classes[predictionIdx]
                
                difSqred = (cur_act_class - cur_algo_pred)**2
                sumDiffsSqrd = sumDiffsSqrd + difSqred
            
            precentCorrect = np.sqrt(sumDiffsSqrd / (len(predictions_On_TestSet)))
            
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
   
        
    def MultiClass_LinReg(self, testDF, trainDF, N_VAL, EP_VAL):
        #Handles Linear Regression for the multi class case
        #Drop the Predictor from the trainDF
        noPred_trainDF = trainDF.drop([self.predictor], axis =1)
        
        #Number of Columns is the _j in Figure 10.6
        numberOfColumns = len(noPred_trainDF.columns)
        
        #Number of Observations is the _t in Figure 10.6
        numberOfObservations = len(noPred_trainDF)
        
        weight_i_List = []
        for classK in range(self.numClassProb):
            #Seed the Inital Weight Values (w_j)
            weight_j_List = []
            for featureIdx in range(numberOfColumns):
                curWeight_J = random.uniform(-0.01, 0.01)
                weight_j_List.append(curWeight_J)
            weight_i_List.append(weight_j_List)
            
        #Convert the List to a Numpy Array
        W_ij_Vector = np.array(weight_i_List, dtype=np.float)
            
 
        #Let the Weights Update Until they Converge
        for curEP in range(EP_VAL):   
            
            #Build the Detlta Weight Inital Array
            delta_weight_i_List = []
            for classK in range(self.numClassProb):
                #Seed the Inital Weight Values (w_j)
                delta_weight_j_List = []
                for featureIdx in range(numberOfColumns):
                    curWeight_J = 0
                    delta_weight_j_List.append(curWeight_J)
                delta_weight_i_List.append(delta_weight_j_List)
            
            #Convert the List to a Numpy Array
            Delta_W_ij_Vector = np.array(delta_weight_i_List, dtype=np.float)
            
            biasTerm = 0
            
            for observationIDx in range(numberOfObservations):
                #Build the X Vector that Represents the Observation
                curObservation = noPred_trainDF.iloc[[observationIDx]]
                X_j_Vector = curObservation.to_numpy()
                X_j_Vector = X_j_Vector[0]
                
                #Get the Actual Classifier
                actual_Y_Class = trainDF[self.predictor].values[observationIDx]
                if (self.probType == 'Regression'):
                    actual_Y_Val = actual_Y_Class
                    
                elif (self.probType == 'Classification'):
                    if(actual_Y_Class == self.zeroClass):
                        actual_Y_Val = 0
                    elif (actual_Y_Class == self.oneClass):
                        actual_Y_Val = 1
                    elif (actual_Y_Class == self.twoClass):
                        actual_Y_Val = 2
                    elif (actual_Y_Class == self.threeClass):
                        actual_Y_Val = 3
                
                
                o_val = np.dot(W_ij_Vector, X_j_Vector) + biasTerm
                
                #Get the Sum of Predidictions
                sumPred = 0
                for classK in range(self.numClassProb):
                    curTerm = math.exp(o_val[classK])
                    sumPred = sumPred + curTerm
                
                #Build the predictions (y_i)
                y_i = []
                for classK in range(self.numClassProb):
                    curTerm = math.exp(o_val[classK])
                    curYI = curTerm / sumPred
                    y_i.append(curYI)
                
                #Update the Updated Weights
                for classK in range(self.numClassProb):
                    Delta_W_ij_Vector =  Delta_W_ij_Vector + (actual_Y_Val - y_i[classK])*X_j_Vector
            
            
            W_ij_Vector = W_ij_Vector + N_VAL*Delta_W_ij_Vector
                
                    
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
                actual_Y_Val = actual_Y_Class
            elif (self.probType == 'Classification'):
                if(actual_Y_Class == self.zeroClass):
                    actual_Y_Val = 0
                elif (actual_Y_Class == self.oneClass):
                    actual_Y_Val = 1
                elif (actual_Y_Class == self.twoClass):
                    actual_Y_Val = 2
                elif (actual_Y_Class == self.threeClass):
                    actual_Y_Val = 3
            
            
            o_val = np.dot(W_ij_Vector, X_j_Vector) + biasTerm
            if (self.probType == 'Regression'):
                prediction_Y_Val = o_val
        
            elif (self.probType == 'Classification'):
                prediction_Y_Val = 1 / (1 + np.exp(-1*o_val))
            
            predictions_On_TestSet.append(prediction_Y_Val)
            
        ####
        #Compare predictions on test set to actual values in the test set
        
        #print(predictions_On_TestSet)
        #Get the Actual Predictions
        actual_Test_Y_Classes = testDF[self.predictor].tolist()
        #print(actual_Test_Y_Classes)
        if (self.probType == 'Regression'):
            precentCorrect = 0 
            sumDiffsSqrd = 0
            for predictionIdx in range(len(predictions_On_TestSet)):
                cur_algo_pred = predictions_On_TestSet[predictionIdx]
                cur_act_class = actual_Test_Y_Classes[predictionIdx]
                
                difSqred = (cur_act_class - cur_algo_pred)**2
                sumDiffsSqrd = sumDiffsSqrd + difSqred
            
            precentCorrect = np.sqrt(sumDiffsSqrd / (len(predictions_On_TestSet)))
            
        elif (self.probType == 'Classification'):
            numCorrect = 0
            for predictionIdx in range(len(predictions_On_TestSet)):
                cur_algo_pred = predictions_On_TestSet[predictionIdx]
                cur_algo_pred = cur_algo_pred.tolist()
                #Apply the soft max
                max_value = max(cur_algo_pred)
                max_index = cur_algo_pred.index(max_value)
                
                
                cur_act_class = actual_Test_Y_Classes[predictionIdx]
                
                if(max_index == 0):
                    cur_algo_pred_class = self.zeroClass
                elif(max_index == 1):
                    cur_algo_pred_class = self.oneClass
                elif(max_index == 2):
                    cur_algo_pred_class = self.twoClass
                elif(max_index == 3):
                    cur_algo_pred_class = self.threeClass
                    
                if(cur_algo_pred_class == cur_act_class):
                    numCorrect = numCorrect + 1
                    
            precentCorrect = numCorrect / (len(predictions_On_TestSet))
            
        return precentCorrect
                
                
            
        
            
            
            
        

        
        
