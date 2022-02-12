# Sarah Wilson
# Project 2 

from typing import List, Tuple, Dict;
import pandas as pd

class DataSet:
    def __init__(self, 
             dataSetName: str = None,
             taskType: str = None,
             dataFilePath: str = None,
             headers: List = None,
             dataTypes: Dict = None,
             missingValueAttributes: List = None,
             #missingValueDataTypeConverters: Dict = None
             ):
        self.name = dataSetName
        self.taskType = taskType
        self.dataFilePath = dataFilePath
        self.headers = headers
        self.dataTypes = dataTypes
        self.missingValueAttributes = missingValueAttributes
        #self.missingValueDataTypeConverters = missingValueDataTypeConverters
        self.rawData = None # Raw Data Frame
        self.rawDataWithDataTypes = None # Data Frame that has been corrected for data types of each Attribute
        self.filledData = None # Filled in Data Frame
        self.finalData = None # Final Data Frame after all needed data clean ups have been applied
        
        #Read in the Raw Data
        self._readInData()
        
        #Apply Data Types for Each Attribute
        self._applyDataTypes()
        
            
    def _readInData(self):
        if ((self.dataFilePath != None) and (self.headers != None)):
            self.rawData = pd.read_csv(self.dataFilePath, names=self.headers)
    
    def _applyDataTypes(self, stringToIntOrNan = True):
        # TODO INSERT Description
        #Build the Dictonary of Converters for the Data Frame Read
        convertersDataType = {}
        if (self.missingValueAttributes != None):
            for attribute in self.missingValueAttributes:
                if (stringToIntOrNan):
                    convertersDataType[attribute] = self._convert_StringToIntOrNaN
                  
        self.rawDataWithDataTypes = pd.read_csv(self.dataFilePath, 
                                                names=self.headers, 
                                                dtype=self.dataTypes, 
                                                converters=convertersDataType)
        
    def _convert_StringToIntOrNaN(self, dataFrameColumn):
        #Converts Strings such as ? that indicate missing data to an Int or Nan
        return pd.to_numeric(dataFrameColumn, errors = 'coerce')
