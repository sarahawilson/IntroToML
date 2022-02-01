import pandas as pd 

class DataHelper(object):
    def __init__(self, dataSetFilePath):
        self.ID = 'Helper_01'
        self.filePath = dataSetFilePath
        self.dataFrame = pd.read_csv(self.filePath)   

    def printData(self):
        print(self.dataFrame)