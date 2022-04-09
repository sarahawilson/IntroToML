# Sarah Wilson 
# 303 - 921 - 7225
# Project 4
# Introduction to Machine Learning

import pandas as pd
import numpy as np
import random 

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
        
     
    def feedforward_prop(self, observation, network):
        #Takes an observation from the data set and passes it through the network
        #layerNameList = list(network.keys())
        layerIdx = 0
        nextLayer_Inputs = observation
        
        for layer in network:
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
            
        
        
        
        
        
        
        
        
        
        