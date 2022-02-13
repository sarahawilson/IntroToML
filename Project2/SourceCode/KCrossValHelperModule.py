# Sarah Wilson 
# 303 - 921 - 7225
# Project 2
# Introduction to Machine Learning

from typing import List, Tuple, Dict
import DataSetHelper
import KNNAlgoHelperModule

class KCrossValHelper:
    def __init__(self,
                 allDataSets: Dict,
                 inputAlgoHelper):
        
        self.name = 'KCrossVal'
        self.numFolds = 5
        self.allDataSets = allDataSets
        self.algoHelper = inputAlgoHelper
        
        
    def createValidation_TuneAndExperimentSets(self):
    # Splits the overall data sets into the 20% that is needed for Validation
    # The other 80% is left for the full algorithm experiment 
        for dataSetName in self.allDataSets: 
            tempCurDataSet = self.allDataSets[dataSetName].finalData.copy(deep=True)
            #Create the 20% Set
            self.allDataSets[dataSetName].finalData_Validation20PercentSet = tempCurDataSet.sample(frac=0.2, random_state=1)
            
            #Create the 80% Set 
            self.allDataSets[dataSetName].finalData_ExperimentSet = tempCurDataSet.drop(self.allDataSets[dataSetName].finalData_Validation20PercentSet.index)
            
    def create_folds(self, inputDataFrame, numFolds):
        # Input data frame
        #   and the number of folds (int) to create out of this data frame
        # Creates the number of disjoint folds from the input data frame
        # Returns a list of these dataframes
        tempInputDataFrame = inputDataFrame.copy(deep=True)
        kFoldDataFramesTest = np.array_split(tempInputDataFrame, numFolds)

        return kFoldDataFramesTest; 
    
    def runKFoldCrossVal(self, printMessage: str):
        print(printMessage)
        for iFoldIndex in range(self.numFolds):
            test =1 
        
    