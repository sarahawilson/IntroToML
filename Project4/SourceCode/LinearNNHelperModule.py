# Sarah Wilson 
# 303 - 921 - 7225
# Project 4
# Introduction to Machine Learning

import pandas as pd
import numpy as np
import random 
import copy

class Linear_NN_Helper:
    def __init__(self, dataSet, numClassProblem, classA, classB):
        self.name = "Linear Neural Network Helper"
        self.dataSetName = dataSet.name
        self.dataSet = dataSet
        self.predictor = dataSet.predictor
        self.probType = dataSet.taskType
        self.numClassProb = numClassProblem
        self.zeroClass = classA
        self.oneClass = classB
        self.NN_Network = []
        
        
    def build_template_network(self, num_input_nodes, num_hidden_layers, num_hidden_nodes, num_output_nodes):
        #Builds the network based off the input dimensions
        network = []
        for layer in range(num_hidden_layers):
            layerName = "HiddenLayer_" + str(layer+1)
            curLayerWeights = {}
            curLayerWeights[layerName] = []
            for hiddenNode in range(num_hidden_nodes):
                #For the number of nodes, and add one additional weight for the bias
                curHiddenNodeWeights = {}
                curHiddenNodeWeights['Weight'] = []
                for inputIdx in range(num_input_nodes + 1):
                    curHiddenNodeWeights['Weight'].append(random.uniform(-0.01, 0.01))
                curLayerWeights[layerName].append(curHiddenNodeWeights)
            network.append(curLayerWeights)
       
        outLayerName = "OutputLayer"
        outputLayerWeights = {}
        outputLayerWeights[outLayerName] = []
        for outputNode in range(num_output_nodes):
            curOutputNodeWeights = {}
            curOutputNodeWeights['Weight'] = []
            for hiddenIdx in range(num_hidden_nodes +1):
                curOutputNodeWeights['Weight'].append(random.uniform(-0.01, 0.01))
            outputLayerWeights[outLayerName].append(curOutputNodeWeights)
        network.append(outputLayerWeights)
        
        self.NN_Network = network
        return network
    
    
    def calcActivation_At_Neuron(self, weightVals: list, inputVals: list):
        #Do the weight sum at each of the Neurons to get the value at
        #that neuron
        
        #Get the Bias
        lengthWeight = len(weightVals);
        biasIdx = lengthWeight-1
        biasTerm = weightVals[biasIdx]
        
        #Convert weight and input to numpy arrays to be able to use dot
        weightValsArray = np.asarray(weightVals);
        #Detele the Bias from this weight list
        weightValsArray = np.delete(weightValsArray,biasIdx)
        
        inputValsArray = np.asarray(inputVals);
        
        #Get the dot product of these arrays
        valueAtNeuron = np.dot(weightValsArray, inputValsArray) + biasTerm
        
        return valueAtNeuron
        
     
    def calcNeuron_output(self, valueAtNeuron):
        #Apply the Sigmoid Function 
        sigmoidResult = 1 / (1 + np.exp(-1*valueAtNeuron))
        return sigmoidResult
      
    def calcSigmoid_derivative(self, neuronOutputVal):
        sigmoidDerivative = neuronOutputVal * (1 - neuronOutputVal)
        return sigmoidDerivative
     
        
    def feedforward_prop(self, observation):
        #Takes an observation from the data set and passes it through the network
        #layerNameList = list(network.keys())
        layerIdx = 0
        nextLayer_Inputs = observation
        
        #for layer in network:
        for layer in self.NN_Network:
            layerNameList = list(layer.keys())
            curLayerName = layerNameList[0]
            layerOuputs = []
            for neuron in layer[curLayerName]:
                valInNeuron = self.calcActivation_At_Neuron(neuron['Weight'], nextLayer_Inputs)
                outputOfNeuron = self.calcNeuron_output(valInNeuron)
                neuron['Output'] = outputOfNeuron
                layerOuputs.append(neuron['Output'])
            layerIdx = layerIdx + 1 
            nextLayer_Inputs = layerOuputs
            
        return nextLayer_Inputs
            
      
    def backwards_prop(self, actualObservationOutput):
        #Calculates the error and backpropigates that through the network
        
        #Flip the Order of the Network to start at the output later
        flippedNetwork = copy.deepcopy(self.NN_Network)
        flippedNetwork.reverse()
        layerIndex = 0
        for layer in flippedNetwork:
            errorList = []
            layerNameList = list(layer.keys())
            curLayerName = layerNameList[0]
            
            #Hit the Output Layer First
            #Step 1 
            if (curLayerName == 'OutputLayer'):
                #Account for there possibly being more that one 
                #output node
                for outNodeIdx in range(len(layer[curLayerName])):
                    neuron = layer[curLayerName][outNodeIdx]
                    if (self.probType == 'Regression'):
                        actual_Predictor = actualObservationOutput
                    elif (self.probType == 'Classification'):
                        if(actualObservationOutput == self.zeroClass):
                            actual_Predictor = 0
                        elif (actualObservationOutput == self.oneClass):
                            actual_Predictor = 1
                    
                    #TODO: Ask Shane how to calcuate the Error for the Output Layer
                    curOutNode_Error = (neuron['Output'] - actual_Predictor)
                    errorList.append(curOutNode_Error)
        
            #Step 3 (yes, really step 3)
            else:
                #Not the output layer but instead a hidden layer
                for hiddenNodeIdx in range(len(layer[curLayerName])):
                    insideError = 0
                    #Get the list of Neurons from the previous layer
                    #if(layerIndex != (len(flippedNetwork)-1)):
                    previousLayer = flippedNetwork[layerIndex -1]
                    layerNameList = list(previousLayer.keys())
                    curPrevLayerName = layerNameList[0]

                        
                    for neuron in previousLayer[curPrevLayerName]:
                        tempWeightArray = np.asarray(neuron['Weight'])
                        tempValueArray = tempWeightArray * neuron['Updated_Weight']
                        culmSumError = np.sum(tempValueArray)
                        insideError = insideError + culmSumError
                    errorList.append(insideError)
            
            #Step 2
            #Done Completetng the Output Layer
            #Need to calcaulte the updated weights to have for the next pervious layer
            for outNodeIdx in range(len(layer[curLayerName])):
                neuron = layer[curLayerName][outNodeIdx]
                outputDerivative = self.calcSigmoid_derivative(neuron['Output'])
                neuron['Updated_Weight'] = errorList[outNodeIdx] *  outputDerivative
            
                        
            layerIndex = layerIndex + 1
        
        #Return the flipped network
        returnNetwork = copy.deepcopy(flippedNetwork)
        returnNetwork.reverse()
        self.NN_Network = returnNetwork
               
    def calc_Updated_Weights(self, observation, NP_Val):
        updatedWeightNetwork = copy.deepcopy(self.NN_Network)
        
        numActWeights = len(observation)
        
        
        for layer in updatedWeightNetwork:
            layerNameList = list(layer.keys())
            curLayerName = layerNameList[0]
            currentInput = []
            #If we aren't the first layer
            #then the input is not the obsevation 
            #but a collection of the previous layers outputs
            if (curLayerName != 'HiddenLayer_1'):
                for neuron in layer[curLayerName]:
                    currentInput.append(neuron['Output'])
            else:
                currentInput = observation

            for neuron in layer[curLayerName]:
                for weightIdx in range(len(currentInput)):
                    #print('PreUpdate')
                    #zz_preUpdateWeight = (neuron['Weight'][weightIdx])
                    neuron['Weight'][weightIdx] = neuron['Weight'][weightIdx] + (NP_Val * neuron['Updated_Weight'] * currentInput[weightIdx])
                    #print('PreUpdate')
                    #zz_postUpdateWeight = (neuron['Weight'][weightIdx])
                #Adjust the Biast Weight
                neuron['Weight'][(numActWeights-1)] = neuron['Weight'][(numActWeights-1)] + (NP_Val * neuron['Updated_Weight'])
                        
        test = 1        
           
      
    def updateWeightsUntilConvergance(self, trainDF, NP_Val, EP_Val):
        
        #Drop the Predictor from the trainDF
        noPred_trainDF = trainDF.drop([self.predictor], axis =1)
        numberOfObservations = len(noPred_trainDF)
                
        for curEp in range(EP_Val):
            #print('epoch')
            #print(str(curEp))
            
            for observationIdx in range(numberOfObservations):
                #print('observation indx')
                #print(str(observationIdx))
                curObservationDF = noPred_trainDF.iloc[[observationIdx]]
                curObservationDF_Array = curObservationDF.to_numpy()
                curObservation = curObservationDF_Array[0]
                
                outputOfUntrainedNN = self.feedforward_prop(curObservation)
                #TODO: This will return a vector of outptus for a multiple class problem
                # figure out how to handle this
                
                actual_Y_Class = trainDF[self.predictor].values[observationIdx]
                self.backwards_prop(actual_Y_Class)
                self.calc_Updated_Weights(curObservation, NP_Val)
    
    def makePrediction(self, curObservationTEST):
        prediction = self.feedforward_prop(curObservationTEST)
        if(self.dataSetName == 'Car Eval'):
            #Take the Prediction with the highest output (softmax)
            prediction = prediction.index(max(prediction))
        return prediction
   
    
    def reportError_LinearNN_withBackProp(self, testDF, trainDF, N_VAL, EP_VAL):
        linNN_Test_Set_Predicitions = self.run_LinearNN_withBackProp(trainDF, testDF, N_VAL, EP_VAL)
        #print(linReg_Test_Set_Predicitions)
        
        
        #Get the Actual Predictions
        actual_Test_Set_Values = testDF[self.predictor].tolist()
        #print(actual_Test_Set_Values)
        
        if (self.probType == 'Regression'):
            precentCorrect = 0 
            sumDiffsSqrd = 0
            for predictionIdx in range(len(linNN_Test_Set_Predicitions)):
                cur_algo_pred = linNN_Test_Set_Predicitions[predictionIdx]
                cur_act_class = actual_Test_Set_Values[predictionIdx]
                
                difSqred = (cur_act_class - cur_algo_pred)**2
                sumDiffsSqrd = sumDiffsSqrd + difSqred
            
            precentCorrect = np.sqrt(sumDiffsSqrd / (len(linNN_Test_Set_Predicitions)))
            
        elif (self.probType == 'Classification'):
            numCorrect = 0
            for predictionIdx in range(len(linNN_Test_Set_Predicitions)):
                cur_algo_pred = linNN_Test_Set_Predicitions[predictionIdx]
                cur_act_class = actual_Test_Set_Values[predictionIdx]
                
                if(cur_algo_pred > 0.5):
                    cur_algo_pred_class = self.oneClass
                else:
                    cur_algo_pred_class = self.zeroClass
                    
                if(cur_algo_pred_class == cur_act_class):
                    numCorrect = numCorrect + 1
                    
            precentCorrect = numCorrect / (len(linNN_Test_Set_Predicitions))
            
        return precentCorrect
          
    def run_LinearNN_withBackProp(self, trainDF, testDF, NP_Val, EP_Val):
        #Runs Linear NN using Back propigation 
        
        numberOfInputs = len(trainDF.columns) - 1
        
        if(self.probType == 'Classification'):
            if(self.dataSetName == 'Car Eval'):
                outputNodes = 4
            else:
                outputNodes = 1
        elif(self.probType == 'Regression'):
            #TODO: I think for regression the number of nodes is probably 1 
            # need to check with Shane
            outputNodes = 1
        
        #Build the Inital Network with 2 hidden layers
        self.build_template_network(numberOfInputs, 2, numberOfInputs, outputNodes)
        
        #Train the Weights until they Converge
        self.updateWeightsUntilConvergance(trainDF, NP_Val, EP_Val)
        
        
        #Start Making Predictions on the test Set
        #Drop the Predictor from the trainDF
        noPred_testDF = testDF.drop([self.predictor], axis =1)
        
        
        numberOfObservations_In_TEST = len(testDF)
        algoPredictions = []
        for obserIdx in range(numberOfObservations_In_TEST):
            curObservationIn_TestDF = noPred_testDF.iloc[[obserIdx]]
            curObservationIn_TestDF_Array = curObservationIn_TestDF.to_numpy()
            curObservationTEST = curObservationIn_TestDF_Array[0]
            cur_algo_pred = self.makePrediction(curObservationTEST)
            cur_algo_pred = cur_algo_pred[0]
            algoPredictions.append(cur_algo_pred)
            
        return algoPredictions
                
            
                    
                    
            
        
        
        
        
        
        
        
        
        
        