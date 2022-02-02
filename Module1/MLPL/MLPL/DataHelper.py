import pandas as pd 
import numpy as np

class DataHelper(object):
    def __init__(self, dataSetFilePath, hasHeader=True):
        self.ID = 'Helper_01'
        self.filePath = dataSetFilePath
        if(hasHeader):
            self.dataFrame = pd.read_csv(self.filePath)
        else:
            self.dataFrame = pd.read_csv(self.filePath, header=None)
            
    def printData(self):
        print(self.dataFrame)
    
    def handleMissingValues(self, missingRepresentation=None):
        #Waiting on Response from discussion question
        #Write a function that, given a dataset, imputes missing
        #values with the feature (column) mean
        #For now assuming that this implies the whole column
        
        #Get the average of the whole column 
        if(missingRepresentation != None): 
            self.dataFrame.replace({missingRepresentation: None}, inplace=True)
            print(self.dataFrame)
            testMeanCol = self.dataFrame.mean(axis=0, skipna=True)
            test2 = self.dataFrame[1].mean()
            print(testMeanCol)
            print(test2)
            print(self.dataFrame.isna())
        
        