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
     
        
    def feedforward_prop(self, observation, network):
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
        
        test =1
        
           
                    
                    
                    
            
        
        
        
        
        
        
        
        
        
        