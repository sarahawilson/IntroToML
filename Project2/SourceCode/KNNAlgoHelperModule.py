# Sarah Wilson 
# 303 - 921 - 7225
# Project 2
# Introduction to Machine Learning

from typing import List, Tuple, Dict

class KNNAlgoHelper:
    def __init__(self,
                 allDataSets: Dict):
        
        self.name = 'KNN'
        self.allDataSets = allDataSets
        self._kNeighbors = None # Can be updated for tuning 
        
        #Convert All Data Sets to Numeric
        self._convertAllDataSetsToNumeric()
        
        
    
    def _convertAllDataSetsToNumeric(self):
        #Take the Nominal Data in the Data Sets and Applies One Hot Encoding
        #Takes the Ordinal Data in the Data Sets and Applies One Hot Encoding
        
        #Define the Tuple for all the data that needs to be one hot encoded
        toApplyOneHotOn =[('Albalone', ['Sex']), 
                        ('Computer Hardware', ['Vendor Name', 'Model Name']),
                        ('Forest Fire', ['month', 'day'])]
        
        
        #TODO: Find out if the Predictor Car Eval needs to be encoded as well
        carEvalOrdinalEncoding = {'Buying': {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1},
                                  'Maint': {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1},
                                  'Lug_Boot': {'big': 3, 'med': 2, 'small': 1},
                                  'Safety': {'high': 3, 'med': 2, 'low': 1}
                                  }

        #Define the Tuple for all the data sets that need to be Ordinal Encoded
        toApplyOrdinalEncodingOn = [('Car Eval',carEvalOrdinalEncoding)
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
        
    def _calculateDistance(self):
        #TODO: Implement 
        test =1           
        
    def runKNNAlgorithm(self):
        #TODO: Implement 
        test = 1