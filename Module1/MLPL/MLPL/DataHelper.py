import pandas as pd 

class DataHelper(object):
    def __init__(self):
        pass
            
    def handleMissingValues(self, inputDataFrame, missingRepresentation=None):
        #Waiting on Response from discussion question
        #Write a function that, given a dataset, imputes missing
        #values with the feature (column) mean
        #For now assuming that this implies the whole column
        #pd.to_numeric(df['s2'], errors='coerce').mean()
        
        #Get the average of the whole column 
        if(missingRepresentation != None): 
            inputDataFrame.replace({missingRepresentation: None}, inplace=True)

            #self.dataFrame.replace({missingRepresentation: None}, inplace=True)
            #self.dataFrame.apply(mean, axis=1)
            
            #print(self.dataFrame)
            #print(self.dataFrame.isna())
            
            #mean = pd.to_numeric(self.dataFrame.iloc[:,1], errors='coerce').mean()
            #print(mean)

            #testMeanCol = self.dataFrame.mean(axis=0, skipna=False)
            #testMeanRow = self.dataFrame.mean(axis=1, skipna=False)

        