#Sarah Wilson 
#Programming Project #1 
#605.649 Introduction to Machine Learning

import pandas as pd
import numpy as np
import copy
from typing import List, Tuple, Dict;

def main(dataFile: str, 
         dataFileHeaders: List = None,
         dataFileDtypes: Dict = None,
         dataFileDtypConvtert = None,
         missingValuesCols: List = None,
         ordinalEncoding: Dict = None,
         nominalOneHotColList: List = None,
         discretizationApply: List = None):
    
    # Dictonary that will store the data frame states as it gets certain types
    # of transformations applied. Will allow for debugging and demonstraitons
    dataProc = {'Raw Data': None,
                 'Raw Data dTypes Applied': None,
                 'Int. Step Average On Features': None,
                 'Filled Data': None,
                 'Ordinal Encoded Data': None,
                 'Nominal One Hot Data': None,
                 'Int. Step Discretization Equal Width': None,
                 'Int. Step Discretization Equal Freq': None,
                 'DataFrame Discretization Equal Width': None,
                 'DataFrame Discretization Equal Freq': None,
                 'Tune and TrainTest Sets': None}
    
    # Get the raw data    
    raw_data = pd.read_csv(dataFile, names=dataFileHeaders)
    dataProc['Raw Data'] = raw_data
    
    #Apply data types to the columns as needed 
    raw_data_typeApplied = pd.read_csv(dataFile, names=dataFileHeaders, dtype=dataFileDtypes, converters=dataFileDtypConvtert)
    dataProc['Raw Data dTypes Applied'] = raw_data_typeApplied
    

    #Apply Transform needed for Missing Attribute Values
    filled_data = raw_data_typeApplied.copy(deep=True)
    if (missingValuesCols != None):
        mean = []
        for col in missingValuesCols:
            mean.append(raw_data_typeApplied[col].mean(axis=0, skipna=True))
        dataProc['Int. Step Average On Features'] = mean
        colIndex = 0
        for col in missingValuesCols:
            filled_data.fillna({col: mean[colIndex]}, inplace=True)
            colIndex = colIndex + 1
        dataProc['Filled Data'] = filled_data

    
    #Apply Transform needed for converting Ordinal Data to Ints
    ordinalEncoded_data = filled_data.copy(deep=True)
    if (ordinalEncoding != None):
        ordinalEncoded_data.replace(to_replace=ordinalEncoding, inplace = True)
        dataProc['Ordinal Encoded Data'] = ordinalEncoded_data
            
    #Apply Transform needed for converting Nominal Data into a One Hot Encoding
    nominalOneHotBase_data = filled_data.copy(deep=True)
    if(nominalOneHotColList != None):
        nominalOneHotEncoded_data = pd.get_dummies(nominalOneHotBase_data, columns = nominalOneHotColList)
        dataProc['Nominal One Hot Data'] = nominalOneHotEncoded_data

        
    #Apply the Discritiztion 
    discretization_data_equalWidth = filled_data.copy(deep=True)
    discretization_data_equalFrequency = filled_data.copy(deep=True)
    if(discretizationApply != None):
        discType = discretizationApply[0]
        numBins = discretizationApply[1]
        discEqualWidthCats = []  #Intermediate Form 
        discEqualFreqCats= []    #Intermediate Form
        for colHeader in discretizationApply[2]:
            if discType == 'Equal Width':
                #Intermediate Form for debug and verification
                discEqualWidthCats.append(pd.cut(discretization_data_equalWidth[colHeader], numBins))
                
                #Apply to the real data frame
                discretization_data_equalWidth[colHeader] = (pd.cut(discretization_data_equalWidth[colHeader], numBins))

                
            if discType == 'Equal Frequency':
                #Intermediate Form for debug and verification
                discEqualFreqCats.append(pd.qcut(discretization_data_equalFrequency[colHeader], numBins, duplicates = 'drop'))
                
                #Apply to the real data frame
                discretization_data_equalFrequency[colHeader] = (pd.qcut(discretization_data_equalFrequency[colHeader], numBins, duplicates = 'drop'))
                               
        #Place data in dictonary to be used in the video demonstration               
        if discType == 'Equal Width':
            dataProc['Int. Step Discretization Equal Width'] = discEqualWidthCats
        if discType == 'Equal Frequnecy':
            dataProc['Int. Step Discretization Equal Freq'] = discEqualFreqCats
                          
    return dataProc
    
    
def convert_StringToIntOrNaN(dataFrameColumn):
    #Converts Strings such as ? that indicate missing data to an Int or Nan
    return pd.to_numeric(dataFrameColumn, errors = 'coerce')

def create_Tune_TrainTest(overallDataFrame):
    tempOverDataSet = overallDataFrame.copy(deep=True)
    #Used to created to seperate data frames (Tune and TrainTest) from the overall data frame
    #Takes the overall data set 
    #Dividies it up into 20%: Tune/Valdiation DataFrame
    #80% that becoems the: TrainTest DataFrame
    data_Tune_TrainTest = {'Tune Data Set': None,
                           'TrainTest Data Set': None}
    
    #Note: This is working under the assumption that the Tune Data Set and the TrainTest Data set must be
    # be comprised of data points that are totally unique. (For example no data point in Tune Data Set)
    # can be found in the TrainTest Data Set
    data_Tune_TrainTest['Tune Data Set'] = tempOverDataSet.sample(frac=0.2, random_state=1)

    #Could uncomment line if the above note does not apply
    #data_Tune_TrainTest['TrainTest Data Set'] = overallDataFrame.sample(frac=0.8, random_state=1)
    data_Tune_TrainTest['TrainTest Data Set'] = tempOverDataSet.drop(data_Tune_TrainTest['Tune Data Set'].index)

    return data_Tune_TrainTest


def create_stratified_folds(inputDataFrame, numFolds):
    # Input data frame
    # and the number of folds (int) to create out of this data frame
    # Creates the number of disjoint folds from the input data frame
    # Returns a list of these dataframes
    tempInputDataFrame = inputDataFrame.copy(deep=True)
    kFoldDataFramesTest = np.array_split(tempInputDataFrame, numFolds)

    return kFoldDataFramesTest;    
    


def cal_mean_std(inputDataFrame, colsApply):
    #Calcualtes the mean and stardand deviation on a certain set of columns 
    # inputDataFrame is the pd.dataframe to get the data from
    # colApply is a list of columns to get the data from in the inputDataFrame
    mean_stdList = []
    for col in colsApply:
        meanCol = inputDataFrame[col].mean(axis=0, skipna=True)
        stdCol = inputDataFrame[col].std(axis=0, skipna=True)
        #If statement here to adddress the divide by zero error in Z Standardization
        if(stdCol == 0.0):
            stdCol = 1.0
        
        meanStdTuple = (col, meanCol, stdCol)
        mean_stdList.append(meanStdTuple)
        
    return mean_stdList


def zStanderdize_data(inputDataFrame, mean_stdList):
    #Applies the Z Standardization to the input data set.
    #On the columns indcaited in the list mean_stdList
    manipulateInputDataFrame = inputDataFrame.copy(deep=True)
    for element in mean_stdList:
        curCol = element[0]
        meanCol = element[1]
        stdCol = element[2]
        manipulateInputDataFrame = manipulateInputDataFrame.apply(lambda dfCol: zStanderdize_ApplyFunction(dfCol, meanCol, stdCol) if dfCol.name == curCol else dfCol)
        
    return manipulateInputDataFrame

def zStanderdize_ApplyFunction(inputDataVaule, mean, std):
    #Describes the z-Standerdize funtion
    zStand = (inputDataVaule - mean)/std;
    return zStand       
    

def runKFold_CrossVal(inputTestTrainDataSetList, taskName=None, classCol=None, zStand=False, zStandHeaders=None, encodedCatType=None):
    numFolds = len(inputTestTrainDataSetList)
    accuracyErrorTupleList = []
    zStandTestTrainDict = {'Train Set': None,
                           'Test Set': None}
    
    print('Running k Fold Cross Validation for task: ' + taskName)
    for iFoldIndex in range(numFolds):
        loopList = copy.deepcopy(inputTestTrainDataSetList)
        testdf = loopList.pop(iFoldIndex)
        traindf = pd.concat(loopList, axis=0)
            
        zStandTestTrainDict['Train Set'] = traindf
        zStandTestTrainDict['Test Set'] = testdf
        
        if(zStand):
            meanStdTuple = cal_mean_std(zStandTestTrainDict['Train Set'], zStandHeaders)
            testdf_zStandTrainSet = zStanderdize_data(zStandTestTrainDict['Train Set'], meanStdTuple)
            testdf_zStandTestSet = zStanderdize_data(zStandTestTrainDict['Test Set'], meanStdTuple)

                    
        if (taskName == 'Classification'):
           accuracyErrorTupleList.append(runSimplePluralityClassAlgo(traindf, testdf, classCol, encodedCatType))
           print('\t Accuracy on fold ' + str(iFoldIndex+1) + ': ' +  str(accuracyErrorTupleList[iFoldIndex][0]))
           print('\t Error on fold ' + str(iFoldIndex+1) + ': ' +  str(accuracyErrorTupleList[iFoldIndex][1]))
           
        if (taskName == 'Regression'):
            accuracyErrorTupleList.append(runSimpleAvgRegAlgo(traindf, testdf, classCol))
            print('\t Accuracy on fold ' + str(iFoldIndex+1) + ': ' +  str(accuracyErrorTupleList[iFoldIndex][0]))
            print('\t Error on fold ' + str(iFoldIndex+1) + ': ' +  str(accuracyErrorTupleList[iFoldIndex][1]))
         
        #End of For Loop clear the dictonary
        zStandTestTrainDict['Train Set'] = None
        zStandTestTrainDict['Test Set'] = None
    
    #Determine average Accuracy and average Error for all folds
    accSum = 0.0
    errSum = 0.0
    for accErrPair in accuracyErrorTupleList:
        curAcc = accErrPair[0]
        accSum = accSum + curAcc
        curErr = accErrPair[1]
        errSum = errSum + curErr
    
    avgAcc = accSum / numFolds
    avgErr = errSum / numFolds
    
    return (avgAcc, avgErr)
    

def runSimplePluralityClassAlgo(trainSet, testSet, classifyOnHeader, encodedCatType=None):
    #This function will run the algoirhtm for a simple pluarity class label
    
    mostCommonInTrain = trainSet[classifyOnHeader].value_counts().idxmax()
    #use this call to account for equal counts. Will return the first hit
    mostCommonInTest = testSet[classifyOnHeader].value_counts().idxmax()
    print('Most Common In Train Set:' + str(mostCommonInTrain))
    print('Most Common in Test Set:' + str(mostCommonInTest))
    
    if (encodedCatType != None):
        if(encodedCatType == 'int'):
            mostCommonInTrain = int(mostCommonInTrain)
            mostCommonInTest = int(mostCommonInTest)
    
    if (mostCommonInTrain == mostCommonInTest):
        err = 0.0
        acc = 1.0
    elif(mostCommonInTrain != mostCommonInTest):
        err = 1.0
        acc = 0.0

    return(acc, err)
    
def runSimpleAvgRegAlgo(trainSet, testSet, regOnHeader):
    #This function will run the algoirhtm for a simple determine the average of the test and train sets
    
    avgInTrain = trainSet[regOnHeader].mean(axis=0, skipna=True)
    avgInTest = testSet[regOnHeader].value_counts().idxmax()
    print('Average In Train Set:' + str(avgInTrain))
    print('Average in Test Set:' + str(avgInTest))
    
    
    err = np.abs(avgInTrain - avgInTest)
    acc = (err ) *100 #TODO: Address divide by zero error

    return(acc, err)
   
if __name__ == "__main__":
    print("MLPL - Machine Learning Pipeline")
    
    ####################
    # BREAST CANCER DATA
    ####################
    breastCancerDataSet_OverallType = 'Classification'
    breastCancerDataSet = r"C:\Users\sarah\Desktop\IntroToML\DataSets\BreastCancer\breast-cancer-wisconsin.data"
    breastCancerHeaders = ['Sample Code Number', 'Clump Thickness', 'Uni. of Cell Size', 'Uni. of Cell Shape', 'Marginal Adhesion', 'Single Ep. Cell Size', 'Bare Nuclei',
                            'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    breastCancerDtypes = {'Sample Code Number': 'int', 'Clump Thickness': 'int', 'Uni. of Cell Size': 'int', 
                          'Uni. of Cell Shape': 'int', 'Marginal Adhesion': 'int', 'Single Ep. Cell Size': 'int', 
                          'Bland Chromatin': 'int', 'Normal Nucleoli': 'int', 'Mitoses': 'int', 
                          'Class': 'int'}
    
    breastCancerDtypeConvterts = {'Bare Nuclei': convert_StringToIntOrNaN}
    breastCancerMissingValCols = ['Bare Nuclei']
    #TODO: This column is an int, the mean is returning a float (?) not sure if this is okay or not
    
    dfSteps_BreastCancer = main(breastCancerDataSet, breastCancerHeaders, breastCancerDtypes, breastCancerDtypeConvterts, breastCancerMissingValCols)
    kFoldListBC = create_stratified_folds(dfSteps_BreastCancer['Filled Data'], 5)
    print('---- BreastCancer Data ----')
    (avgAcc, avgErr) = runKFold_CrossVal(kFoldListBC, 'Classification', 'Class', False, None, 'int')
    print('Average Accuracy of Simple Pluarity Predictor:' + str(avgAcc))
    print('Average Absolute Error of Simple Pluarity Predictor:' + str(avgErr))
    print('---- END BreastCancer Data ----')
    
    ####################
    # CAR DATA
    ####################
    carEvalDataSet_OverallType = 'Classification'
    carEvalDataSet = r"C:\Users\sarah\Desktop\IntroToML\DataSets\CarEvaluation\car.data"
    carEvalHeaders = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_Boot', 'Safety', 'Car Acceptability']
    carDtypeDict = {'Buying': 'str', 'Maint': 'str', 'Lug_Boot': 'str', 'Safety': 'str', 'Car Acceptability': 'str'}
    carDtypeConvterts = {'Doors': convert_StringToIntOrNaN, 'Persons': convert_StringToIntOrNaN}
    #TODO: handle the 5-more (just make value 5 for doors)
    #TODO: handle the more (just make the value max+1 for persons)
    
    dfSteps_CarEval = main(carEvalDataSet, carEvalHeaders, carDtypeDict, carDtypeConvterts)
    print('---- CarEval Data ----')
    kFoldListCE = create_stratified_folds(dfSteps_CarEval['Raw Data dTypes Applied'], 5)
    (avgAcc, avgErr) = runKFold_CrossVal(kFoldListCE, 'Classification', 'Car Acceptability')
    print('Average Accuracy of Simple Pluarity Predictor:' + str(avgAcc))
    print('Average Absolute Error of Simple Pluarity Predictor:' + str(avgErr))
    print('---- END CarEval Data ----')
 
    
    ####################
    # CONGRESSIONAL VODE DATA
    ####################
    congVoteDataSet_OverallType = 'Classification'
    congVoteDataSet = r"C:\Users\sarah\Desktop\IntroToML\DataSets\CongressionalVote\house-votes-84.data"
    congVoteHeaders = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                       'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
                       'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
    
    dfSteps_CongVote = main(congVoteDataSet, congVoteHeaders)
    kFoldListCV = create_stratified_folds(dfSteps_CongVote['Raw Data dTypes Applied'], 5)
    print('---- Congressional Vote Data ----')
    (avgAcc, avgErr) = runKFold_CrossVal(kFoldListCV, 'Classification', 'Class Name')
    print('Average Accuracy of Simple Pluarity Predictor:' + str(avgAcc))
    print('Average Absolute Error of Simple Pluarity Predictor:' + str(avgErr))
    print('---- END Congressional Vote Data ----')   
    
    
    ####################
    # COMPUTER HARDWARE DATA
    ####################
    compHardwareDataSet_OverallType = 'Regression'
    compHardwareDataSet = r"C:\Users\sarah\Desktop\IntroToML\DataSets\ComputerHardware\machine.data"
    compHardwareHeader = ['Vendor Name','Model Name','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
    compHardwareDtypeDict = {'Vendor Name': 'str','Model Name': 'str','MYCT': 'int','MMIN': 'int',
                             'MMAX': 'int','CACH': 'int','CHMIN': 'int','CHMAX': 'int','PRP': 'int','ERP': 'int'}
    
    dfSteps_CompHW = main(compHardwareDataSet, compHardwareHeader, compHardwareDtypeDict)
    kFoldListCHW = create_stratified_folds(dfSteps_CompHW['Raw Data dTypes Applied'], 5)
    print('---- Computer Hardware Data ----')
    (avgAcc, avgErr) = runKFold_CrossVal(kFoldListCHW, 'Regression', 'PRP')
    print('Average Absolute Error of Simple Mean Predictor:' + str(avgErr))
    print('---- END Computer Hardware Data ----')   
    
     ####################
    # ALBALONDE DATA
    ####################
    abaloneDataSet_OverallType = 'Regression'
    abaloneDataSet = r"C:\Users\sarah\Desktop\IntroToML\DataSets\Abalone\abalone.data"
    abaloneHeaders = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
    abaloneDtypeDict = {'Sex': 'str', 'Length': 'float' , 'Diameter': 'float', 'Height': 'float', 
                        'Whole Weight':'float', 'Shucked Weight':'float', 'Viscera Weight': 'float', 'Shell Weight': 'float', 'Rings': 'int'}
    
    dfSteps_Abalone = main(abaloneDataSet, abaloneHeaders, abaloneDtypeDict)
    kFoldListABL = create_stratified_folds(dfSteps_Abalone['Raw Data dTypes Applied'], 5)
    print('---- Alablone Data ----')
    (avgAcc, avgErr) = runKFold_CrossVal(kFoldListABL, 'Regression', 'Rings')
    print('Average Absolute Error of Simple Mean Predictor:' + str(avgErr))
    print('---- END Alablone Data ----')   
    

    ####################
    # FOREST FIRE DATA
    ####################
    forestFireDataSet_OverallType = 'Regression'
    forestFireDataSet = r"C:\Users\sarah\Desktop\IntroToML\DataSets\ForestFires\forestfires.data"
    
    dfSteps_ForestFire = main(forestFireDataSet)
    kFoldListFORFIRE = create_stratified_folds(dfSteps_ForestFire['Raw Data dTypes Applied'], 5)
    print('---- Forest Fire Data ----')
    (avgAcc, avgErr) = runKFold_CrossVal(kFoldListFORFIRE, 'Regression', 'area')
    print('Average Absolute Error of Simple Mean Predictor:' + str(avgErr))
    print('---- END Forest Fire Data ----')   
    
    
    #Tests!
    #This section can be uncommented for debugging and demonstration
    #This is what was used during the video presentaiton that was submitted
    ####################
    # EXAMPLE TEST SETS DATA
    ####################
#    #This Test Set shows that the missing values for an Attribute are being replaeced with the mean of all Data Points for that Attribute
#    testAvgDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\colAvg.data"
#    testAvgDataHeaders = ['Col0','Col1','Col2']
#    testAvgDtypeConvterts = {'Col0': convert_StringToIntOrNaN, 'Col2': convert_StringToIntOrNaN}
#    testAvgMissingValCols = ['Col0', 'Col2']
#    
#    
#    #This Test Set shows that Ordinal Data that comes in as strings, is having it's order perseved through ints 
#    testOrdinalDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\ordinal.data"
#    testOrdinalHeaders = ['Degree','Val1','Val2']
#    testOrdinalDtypeDict = {'Degree': 'str','Val1': 'int','Val2': 'int'}
#    
#    testOrdinalEncodingDict = {'Degree': {'HighSchool': '0', 'Bachelors': '1', 'Graduate': '2'}}
#    #Nested Dictornary {'ColHeaderName' : {Encoding Dict}, ect ect}
#    
#    #This Test set shows that the Nominal Data that comes in as strings, is having a one hot encoding applied
#    testNomOneHotDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\nominalOneHot.data"
#    testNomOneHotlHeaders = ['Color','Val1','Val2']
#    testNomOneHotDtypeDict = {'Color': 'str','Val1': 'int','Val2': 'int'}
#    testNomOneHotColList = ['Color']
#    
#    #This Test sets shows Even Width Discretizaiton 
#    testDiscEWDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\discEvenWidth.data"
#    testDiscEWHeaders = ['EWCol','Val1','Val2']
#    #DiscList format is Type of Disc / Number of Bins /  List Column to apply on
#    testDiscEWDiscList = ['Equal Width', 10, ['EWCol']]
#    
#    #This Test set shows Even Frequency Discretizaiton
#    testDiscEFDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\discEvenFreq.data"
#    testDiscEFHeaders = ['EFCol','Val1','Val2']
#    #DiscList format is Type of Disc / Number of Bins /  List Column to apply on
#    testDiscEFDiscList = ['Equal Frequency', 3, ['EFCol']]
#    
#    #This Test Set shows z-Standerdization Being Applied
#    testZStandDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\zStanderdization.data"
#    testZStandFHeaders = ['Val0','Val1','Val2']
#    
#    #This is a test of k-Fold Cross Validation on an simple Classification Data Set 
#    kFClassTestDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\kFold_Classification.data"
#    kFClassTestHeaders = ['Risk', 'Debt']
    
#    dfSteps_testAvg = main(testAvgDataSet, testAvgDataHeaders, None, testAvgDtypeConvterts, testAvgMissingValCols)
#    dfSteps_testOrdinal = main(testOrdinalDataSet, testOrdinalHeaders, testOrdinalDtypeDict, None, None, testOrdinalEncodingDict)
#    dfSteps_testNomOneHot = main(testNomOneHotDataSet, testNomOneHotlHeaders, testNomOneHotDtypeDict, None, None, None, testNomOneHotColList)
#    dfSteps_testDicEvenWidth = main(testDiscEWDataSet, testDiscEWHeaders, None, None, None, None, None, testDiscEWDiscList)
#    dfSteps_testDicEvenFreq = main(testDiscEFDataSet, testDiscEFHeaders, None, None, None, None, None, testDiscEFDiscList)
#    dfSteps_testZStand = main(testZStandDataSet, testZStandFHeaders)
#    dfSteps_testkFoldClass = main(kFClassTestDataSet, kFClassTestHeaders)
#    
#    #Tune and TrainTest Data Sets from the ZStandard DataFrame
#    dictTuneTestTrainZStand = create_Tune_TrainTest(dfSteps_testZStand['Raw Data dTypes Applied'])
#    meanStdTrainZStand = cal_mean_std(dictTuneTestTrainZStand['TrainTest Data Set'], testZStandFHeaders)
#    standerdizedZStandTrainTestSet = zStanderdize_data(dictTuneTestTrainZStand['TrainTest Data Set'], meanStdTrainZStand)
#    standerdizedZStandTunetSet = zStanderdize_data(dictTuneTestTrainZStand['Tune Data Set'], meanStdTrainZStand)
#    
#    #Show 5 folds being built
#    kFoldsExampleList = create_stratified_folds(dfSteps_testkFoldClass['Raw Data dTypes Applied'], 5)
#    
#    print('Demo z Standard---')
#    (avgAcc, avgErr) = runKFold_CrossVal(kFoldsExampleList, 'Regression', 'Debt', True, ['Debt'], True)
#    
#    print('---Classification---')
#    (avgAcc, avgErr) = runKFold_CrossVal(kFoldsExampleList, 'Classification', 'Risk')
#    print('Average Accuracy of Simple Pluarity Predictor:' + str(avgAcc))
#    print('Average Absolute Error of Simple Pluarity Predictor:' + str(avgErr))
#    
#    print('---Regression---')
#    (avgAcc, avgErr) = runKFold_CrossVal(kFoldsExampleList, 'Regression', 'Debt')
#    print('Average Accuracy of Simple Average Predictor:' + str(avgAcc))
#    print('Average Absolute Error of Simple Average Predictor:' + str(avgErr))
#    #print(avgErr)


    