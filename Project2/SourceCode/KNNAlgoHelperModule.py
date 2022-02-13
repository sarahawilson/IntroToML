# Sarah Wilson 
# 303 - 921 - 7225
# Project 2
# Introduction to Machine Learning

from typing import List, Tuple, Dict
import DataSetHelper 

class KNNAlgoHelper:
    def __init__(self,
                 allDataSets: Dict):
        
        self.name = 'KNN'
        self.allDataSets = allDataSets
    
    def _convertAllDataSetsToNumberic(self):
        #TODO: Insert descriptions
        
        #Define the Tuple for all the data that needs to be one hot encoded
        toApplyOneHotOn =[('Albalone', ['Sex']), 
                        ('Computer Hardware', ['Vendor Name', 'Model Name']),
                        ('Forest Fire', ['month', 'day'])]
        
        #Congressional Vote
        # Skipping for now need to figure out 
        
        
        for dataSet in allDataSets:
        # Need to ask all categorital data to be 
        # converted to integers 