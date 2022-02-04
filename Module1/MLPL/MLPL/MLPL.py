#Sarah Wilson 

import DataHelper
import pandas as pd
from typing import List, Tuple, Dict;

def main(dataFile: str, 
         dataFileHeaders: List = None,
         dataFileDtypes: Dict = None,
         dataFileDtypConvtert = None,
         missingValuesCols: List = None,
         ordinalEncoding: Dict = None,
         nominalOneHotColList: List = None):
    
    dataSteps = []  #List that will store the steps as the data gets prepped for processing
    #TODO: Update this to a dictonary
    
    raw_data = pd.read_csv(dataFile, names=dataFileHeaders)
    dataSteps.append(raw_data)
    
    raw_data_typeApplied = pd.read_csv(dataFile, names=dataFileHeaders, dtype=dataFileDtypes, converters=dataFileDtypConvtert)
    dataSteps.append(raw_data_typeApplied)
    
    filled_data = raw_data_typeApplied.copy(deep=True)
    
    #Apply Transform needed for Missing Attribute Values
    if (missingValuesCols != None):
        mean = []
        for col in missingValuesCols:
            mean.append(raw_data_typeApplied[col].mean(axis=0, skipna=True))
        dataSteps.append(mean)
        colIndex = 0
        for col in missingValuesCols:
            filled_data.fillna({col: mean[colIndex]}, inplace=True)
            colIndex = colIndex + 1
        dataSteps.append(filled_data)
    else:
        dataSteps.append('No Missing Values: Mean NA')
        dataSteps.append(filled_data)
    

    ordinalEncoded_data = filled_data.copy(deep=True)
    #Apply Transform needed for converting Ordinal Data to Ints   
    if (ordinalEncoding != None):
        ordinalEncoded_data.replace(to_replace=ordinalEncoding, inplace = True)
        #for col_header, encoding in ordinalEncoding.items():
            #ordinalEncoded_data
        dataSteps.append(ordinalEncoded_data)
    else:
        #This is just to keep the overall list structure in tact (Might want to make this a dict)
        dataSteps.append(ordinalEncoded_data)
            
    #nominalOneHotEncoded_data = ordinalEncoded_data.copy(deep=True)  
    if(nominalOneHotColList != None):
        nominalOneHotEncoded_data = pd.get_dummies(filled_data, columns = nominalOneHotColList)
        dataSteps.append(nominalOneHotEncoded_data)
    else:
        dataSteps.append(ordinalEncoded_data)
        
        
    return dataSteps
    
    #categoial data on the mean -> if it's a stirng / yes or no just use the most coming occuring value

  
def convert_StringToIntOrNaN(dataFrameColumn):
    return pd.to_numeric(dataFrameColumn, errors = 'coerce')

    
if __name__ == "__main__":
    print("MLPL - Machine Learning Pipeline")
    
    
    ####################
    # ALBALONDE DATA
    ####################
    abaloneDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\Abalone\abalone.data"
    abaloneHeaders = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
    abaloneDtypeDict = {'Sex': 'str', 'Length': 'float' , 'Diameter': 'float', 'Height': 'float', 
                        'Whole Weight':'float', 'Shucked Weight':'float', 'Viscera Weight': 'float', 'Shell Weight': 'float', 'Rings': 'int'}


    ####################
    # BREAST CANCER DATA
    ####################
    breastCancerDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\BreastCancer\breast-cancer-wisconsin.data"
    breastCancerHeaders = ['Sample Code Number', 'Clump Thickness', 'Uni. of Cell Size', 'Uni. of Cell Shape', 'Marginal Adhesion', 'Single Ep. Cell Size', 'Bare Nuclei',
                            'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    breastCancerDtypes = {'Sample Code Number': 'int', 'Clump Thickness': 'int', 'Uni. of Cell Size': 'int', 
                          'Uni. of Cell Shape': 'int', 'Marginal Adhesion': 'int', 'Single Ep. Cell Size': 'int', 
                          'Bland Chromatin': 'int', 'Normal Nucleoli': 'int', 'Mitoses': 'int', 
                          'Class': 'int'}
    
    breastCancerDtypeConvterts = {'Bare Nuclei': convert_StringToIntOrNaN}
    breastCancerMissingValCols = ['Bare Nuclei']
    #TODO: This column is an int, the mean is returning a float (?) not sure if this is okay or not
    
    
    ####################
    # CAR DATA
    ####################
    carEvalDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\CarEvaluation\car.data"
    carEvalHeaders = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_Boot', 'Safety', 'Car Acceptability']
    carDtypeDict = {'Buying': 'str', 'Maint': 'str', 'Lug_Boot': 'str', 'Safety': 'str', 'Car Acceptability': 'str'}
    carDtypeConvterts = {'Doors': convert_StringToIntOrNaN, 'Persons': convert_StringToIntOrNaN}
    #TODO: handle the 5-more (just make value 5 for doors)
    #TODO: handle the more (just make the value max+1 for persons)
    
    
    ####################
    # COMPUTER HARDWARE DATA
    ####################
    compHardwareDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\ComputerHardware\machine.data"
    compHardwareHeader = ['Vendor Name','Model Name','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
    compHardwareDtypeDict = {'Vendor Name': 'str','Model Name': 'str','MYCT': 'int','MMIN': 'int',
                             'MMAX': 'int','CACH': 'int','CHMIN': 'int','CHMAX': 'int','PRP': 'int','ERP': 'int'}
    
    
    ####################
    # CONGRESSIONAL VODE DATA
    ####################
    congVoteDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\CongressionalVote\house-votes-84.data"
    congVoteHeaders = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                       'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
                       'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
    #TODO: This data seems like it would be a good candiate for the 1 hot encoding
    #basically there is Voted Yes / Voted No / Netural 
    #YES = 100
    #No = 010
    #Neautral = 000
    
    
    ####################
    # FOREST FIRE DATA
    ####################
    forestFireDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\ForestFires\forestfires.data"
    
    
    ####################
    # EXAMPLE TEST SETS DATA
    ####################
    #This Test Set shows that the missing values for an Attribute are being replaeced with the mean of all Data Points for that Attribute
    testAvgDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\colAvg.data"
    testAvgDataHeaders = ['Col0','Col1','Col2']
    testAvgDtypeConvterts = {'Col0': convert_StringToIntOrNaN, 'Col2': convert_StringToIntOrNaN}
    testAvgMissingValCols = ['Col0', 'Col2']
    
    
    #This Test Set shows that Ordinal Data that comes in as strings, is having it's order perseved through ints 
    testOrdinalDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\ordinal.data"
    testOrdinalHeaders = ['Degree','Val1','Val2']
    testOrdinalDtypeDict = {'Degree': 'str','Val1': 'int','Val2': 'int'}
    
    testOrdinalEncodingDict = {'Degree': {'HighSchool': '0', 'Bachelors': '1', 'Graduate': '2'}}
    #Nested Dictornary {'ColHeaderName' : {Encoding Dict}, ect ect}
    
    #This Test set shows that the Nominal Data that comes in as strings, is having a one hot encoding applied
    testNomOneHotDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\nominalOneHot.data"
    testNomOneHotlHeaders = ['Color','Val1','Val2']
    testNomOneHotDtypeDict = {'Color': 'str','Val1': 'int','Val2': 'int'}
    testNomOneHotColList = ['Color']
    
    
    dfSteps_Abalone = main(abaloneDataSet, abaloneHeaders, abaloneDtypeDict)
    dfSteps_BreastCancer = main(breastCancerDataSet, breastCancerHeaders, breastCancerDtypes, breastCancerDtypeConvterts, breastCancerMissingValCols)
    dfSteps_CarEval = main(carEvalDataSet, carEvalHeaders, carDtypeDict, carDtypeConvterts)
    dfSteps_CompHW = main(compHardwareDataSet, compHardwareHeader, compHardwareDtypeDict)
    dfSteps_CongVote = main(congVoteDataSet, congVoteHeaders)
    dfSteps_ForestFire = main(forestFireDataSet)
    
    #Tests!
    dfSteps_testAvg = main(testAvgDataSet, testAvgDataHeaders, None, testAvgDtypeConvterts, testAvgMissingValCols)
    dfSteps_testOrdinal = main(testOrdinalDataSet, testOrdinalHeaders, testOrdinalDtypeDict, None, None, testOrdinalEncodingDict)
    dfSteps_testNomOneHot = main(testNomOneHotDataSet, testNomOneHotlHeaders, testNomOneHotDtypeDict, None, None, None, testNomOneHotColList)
    
    
    


    