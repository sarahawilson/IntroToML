# Sarah Wilson 
# 303 - 921 - 7225
# Project 2
# Introduction to Machine Learning

from typing import List, Tuple, Dict;
import pandas as pd
import copy

class DataSet:
    def __init__(self, 
             dataSetName: str = None,
             taskType: str = None,
             predictor: str = None,
             dataFilePath: str = None,
             headers: List = None,
             dataTypes: Dict = None,
             missingValueAttributes: List = None,
             applyConversionValueAttribues: Tuple = None
             ):
        self.name = dataSetName
        self.taskType = taskType
        self.predictor = predictor
        self.dataFilePath = dataFilePath
        self.headers = headers
        self.dataTypes = dataTypes
        self.missingValueAttributes = missingValueAttributes
        self.applyConversionValueAttributes = applyConversionValueAttribues
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
            self.finalData = pd.get_dummies(self.finalData, columns = oneHotAttributes)