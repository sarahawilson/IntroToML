# Sarah Wilson 
# 303 - 921 - 7225
# Project 2
# Introduction to Machine Learning

from typing import List, Tuple, Dict;
import pandas as pd

class DataSet:
    def __init__(self, 
             dataSetName: str = None,
             taskType: str = None,
             predictor: str = None,
             dataFilePath: str = None,
             headers: List = None,
             dataTypes: Dict = None,
             missingValueAttributes: List = None,
             applyConversionValueAttribues: Tuple = None,
             id3ColTypes: Dict = None
             ):
        self.name = dataSetName
        self.taskType = taskType
        self.predictor = predictor
        self.dataFilePath = dataFilePath
        self.headers = headers
        self.dataTypes = dataTypes
        self.missingValueAttributes = missingValueAttributes
        self.applyConversionValueAttributes = applyConversionValueAttribues
        self.id3ColTypes = id3ColTypes
        self.rawData = None # Raw Data Frame
        self.rawDataWithDataTypes = None # Data Frame that has been corrected for data types of each Attribute
        self.finalData = None # Final Data Frame after all needed data clean ups have been applied
        
        self.finalData_Validation20PercentSet = None
        self.finalData_ExperimentSet = None
        
        self.finalData_TestSet = None
        self.finalData_TrainSet = None
        
        #Read in the Raw Data
        self._readInData()
        
        #Fill in missing data and apply approritate data types for each attribute
        self._fillMissingAndApplyTypesToData()
        
        #Fill in the Missing Value with the average of the Attribute 
        self._fillMissingValueWithMeanOfAttribute()
        
        #Adjust the data (such as 5more, more) to be integer values
        self._fillWithAdjustedDataValues()
        
        #Generate the final data set to be used in additional algorithms
        self._generateFinalDataSet()
        
            
    def _readInData(self):
        if ((self.dataFilePath != None) and (self.headers != None)):
            self.rawData = pd.read_csv(self.dataFilePath, names=self.headers)
    
    def _fillMissingAndApplyTypesToData(self):
        #TODO Insert Description
        convertersMapping = {}
        if (self.missingValueAttributes != None):
            for attribute in self.missingValueAttributes:
                    convertersMapping[attribute] = self._convert_StringToNaN
        
        elif((self.missingValueAttributes == None) and (self.applyConversionValueAttributes != None)):
            for attributePair in self.applyConversionValueAttributes:
                curAttribute = attributePair[0]
                convertersMapping[curAttribute] = self._convert_StringToNaN
            
               
        self.rawDataWithDataTypes = pd.read_csv(self.dataFilePath, 
                                                names=self.headers, 
                                                dtype=self.dataTypes, 
                                                converters=convertersMapping)


    def _convert_StringToNaN(self, dataFrameColumn):
        #Converts Strings such as ? that indicate missing data to an Int or Nan
        return pd.to_numeric(dataFrameColumn, errors = 'coerce')           
     
        
    def _fillMissingValueWithMeanOfAttribute(self):
        if (self.missingValueAttributes != None):
            mean = []
            for attribute in self.missingValueAttributes:
                mean.append(self.rawDataWithDataTypes[attribute].mean(axis=0, skipna=True))
            
            attIndex = 0
            for attribute in self.missingValueAttributes:
                self.rawDataWithDataTypes.fillna({attribute: mean[attIndex]}, inplace=True)
            attIndex = attIndex + 1
        
        
    def _fillWithAdjustedDataValues(self):
        if(self.applyConversionValueAttributes != None):
            for attribuePair in self.applyConversionValueAttributes:
                curAttribute = attribuePair[0]
                curReplaceVal = attribuePair[1]
                self.rawDataWithDataTypes.fillna({curAttribute: curReplaceVal}, inplace=True)
                
    def _generateFinalDataSet(self):
        self.finalData = self.rawDataWithDataTypes.copy(deep=True)


    def applyOneHotEncoding(self,
                            oneHotAttributes: List = None):
        
        if(oneHotAttributes != None):
            self.finalData = pd.get_dummies(self.finalData, columns = oneHotAttributes,  dtype=int)
            
            
            
            
def ConvertDataSetsToNumeric(allDataSets: Dict):
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
                                  #'Car Acceptability': {'vgood': 4, 'good': 3, 'acc': 2, 'unacc': 1}
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
        
    # Loop over the data sets and apply the 
    # One hot encoding to those that need it 
    # Based on the toApplyOneHotOn Tuple above
    for dataSetName in allDataSets:
        for curTuple in toApplyOneHotOn:
            applyOnDataSetName = curTuple[0]
            if (applyOnDataSetName == dataSetName):
                allDataSets[dataSetName].applyOneHotEncoding(curTuple[1])
                break

        #Now apply the Ordinal Data Encoding
        for curOrdTuple in toApplyOrdinalEncodingOn:
            applyOrdOnDataSetName = curOrdTuple[0]
            if (applyOrdOnDataSetName == dataSetName):
                allDataSets[dataSetName].finalData.replace(to_replace=curOrdTuple[1], inplace = True)    
                
    return allDataSets


def NormailzeDataSets(allDataSets: Dict):
    #Loop over all the data sets
    #scale the data between 1 and 0
    for dataSetName in allDataSets:
        newFinalData = allDataSets[dataSetName].finalData.copy(deep=True)
        for featureName in allDataSets[dataSetName].finalData:
            #Skip the Classifier 
            if(featureName == allDataSets[dataSetName].predictor):
                continue
            maxColVal = allDataSets[dataSetName].finalData[featureName].max()
            minColVal = allDataSets[dataSetName].finalData[featureName].min()
            divZeroCheck = (maxColVal - minColVal)
            if (divZeroCheck == 0):
                print('Division by Zero Encountered!')
            else:
                newFinalData[featureName] = (allDataSets[dataSetName].finalData[featureName] - minColVal) / (maxColVal - minColVal)
                
        allDataSets[dataSetName].finalData = newFinalData
        
    return allDataSets
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
