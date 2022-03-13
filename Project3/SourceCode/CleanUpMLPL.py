# Sarah Wilson 
# 303 - 921 - 7225
# Project 3
# Introduction to Machine Learning

from typing import List, Tuple, Dict;
import DataSetHelper 
import KCrossValHelperModule
import ID3HelperModule


def defineAllDataSets()->Dict:
    # A Dictonary that holds each data set as an object that can be used 
    # in futher processing.
    # Key: Data Set Name
    # Value: Instance of Data Set Object 
    allDataSetObjects = {}
    
    ####################
    # BREAST CANCER DATA
    ####################
    # Define the needed variables
    breastCancerName = 'Breast Cancer'
    breastCancerDataSet_OverallType = 'Classification'
    breastCancerDataSet_Predictor = 'Class'
    breastCancerDataSetPath = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\BreastCancer\breast-cancer-wisconsin.data"
    breastCancerHeaders = ['Sample Code Number', 'Clump Thickness', 'Uni. of Cell Size', 'Uni. of Cell Shape', 
                           'Marginal Adhesion', 'Single Ep. Cell Size', 'Bare Nuclei',
                            'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    breastCancerDtypes = {'Sample Code Number': 'int', 'Clump Thickness': 'int', 'Uni. of Cell Size': 'int', 
                          'Uni. of Cell Shape': 'int', 'Marginal Adhesion': 'int', 'Single Ep. Cell Size': 'int', 
                          'Bland Chromatin': 'int', 'Normal Nucleoli': 'int', 'Mitoses': 'int', 
                          'Class': 'int'}
    
    
    breastCancerID3Types = {'Sample Code Number': 'Num', 'Clump Thickness': 'Num', 'Uni. of Cell Size':'Num', 'Uni. of Cell Shape': 'Num', 
                           'Marginal Adhesion': 'Num', 'Single Ep. Cell Size': 'Num', 'Bare Nuclei': 'Num',
                            'Bland Chromatin': 'Num', 'Normal Nucleoli': 'Num', 'Mitoses': 'Num', 'Class': 'Num'}
    
    #breastCancerDtypeConvterts = {'Bare Nuclei': convert_StringToIntOrNaN}
    breastCancerMissingValAttributes = ['Bare Nuclei']
    breastCancerApplyConversionAttributes = None
    #TODO: This column is an int, the mean is returning a float (?) not sure if this is okay or not
    
    #Create an Instance of the Data Set Class 
    breastCancerDS = DataSetHelper.DataSet(breastCancerName, 
                             breastCancerDataSet_OverallType,
                             breastCancerDataSet_Predictor,
                             breastCancerDataSetPath,
                             breastCancerHeaders,
                             breastCancerDtypes,
                             breastCancerMissingValAttributes,
                             breastCancerApplyConversionAttributes,
                             breastCancerID3Types
                             )
    
    #Add to the Overall Data Set Dictonary 
    allDataSetObjects[breastCancerName] = breastCancerDS
    
    
    ####################
    # CAR DATA
    ####################
    # Define the needed variables
    carEvalDataSetName = 'Car Eval'
    carEvalDataSet_OverallType = 'Classification'
    carEvalDataSet_Predictor = 'Car Acceptability'
    carEvalDataSetPath = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\CarEvaluation\car.data"
    carEvalHeaders = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_Boot', 'Safety', 'Car Acceptability']
    carDtypes = {'Buying': 'str', 'Maint': 'str', 'Lug_Boot': 'str', 'Safety': 'str', 'Car Acceptability': 'str'}
    carMissingValAttributes = None
    carApplyConversionValueAttribues = [('Doors', 5), ('Persons', 5)]
    carEvalID3Types = {'Buying': 'Cat', 'Maint': 'Cat', 'Doors': 'Num', 'Persons': 'Num', 'Lug_Boot': 'Cat', 'Safety': 'Cat', 'Car Acceptability': 'Cat'}
    
    #Create an Instance of the Data Set Class
    carEvalDS = DataSetHelper.DataSet(carEvalDataSetName,
                                      carEvalDataSet_OverallType,
                                      carEvalDataSet_Predictor,
                                      carEvalDataSetPath,
                                      carEvalHeaders,
                                      carDtypes,
                                      carMissingValAttributes,
                                      carApplyConversionValueAttribues,
                                      carEvalID3Types
                                      )
    
    #Add to the Overall Data Set Dictonary 
    allDataSetObjects[carEvalDataSetName] = carEvalDS
    
    ####################
    # CONGRESSIONAL VODE DATA
    ####################
    congVoteDataSetName = "Congressional Vote"
    congVoteDataSet_OverallType = 'Classification'
    congVoteDataSet_Predictor = 'Class Name'
    congVoteDataSetPath = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\CongressionalVote\house-votes-84.data"
    congVoteHeaders = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                       'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
                       'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
    congVoteID3Types = {'Class Name': 'Cat', 'handicapped-infants': 'Cat', 'water-project-cost-sharing': 'Cat', 'adoption-of-the-budget-resolution': 'Cat', 'physician-fee-freeze': 'Cat', 'el-salvador-aid': 'Cat',
                       'religious-groups-in-schools': 'Cat', 'anti-satellite-test-ban': 'Cat', 'aid-to-nicaraguan-contras': 'Cat', 'mx-missile': 'Cat', 'immigration': 'Cat', 'synfuels-corporation-cutback': 'Cat',
                       'education-spending': 'Cat', 'superfund-right-to-sue': 'Cat', 'crime': 'Cat', 'duty-free-exports': 'Cat', 'export-administration-act-south-africa': 'Cat'}
    
    #Create an Instance of the Data Set Class
    congVoteDS = DataSetHelper.DataSet(congVoteDataSetName,
                                       congVoteDataSet_OverallType,
                                       congVoteDataSet_Predictor,
                                       congVoteDataSetPath,
                                       congVoteHeaders,
                                       None,
                                       None,
                                       None,
                                       congVoteID3Types)
    
    #Add to the Overall Data Set Dictonary 
    allDataSetObjects[congVoteDataSetName] = congVoteDS
    
    
    ####################
    # COMPUTER HARDWARE DATA
    ####################
    compHardwareDataSetName = 'Computer Hardware'
    compHardwareDataSet_OverallType = 'Regression'
    compHardwareDataSet_Predictor = 'PRP'
    compHardwareDataSetPath = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\ComputerHardware\machine.data"
    compHardwareHeader = ['Vendor Name','Model Name','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
    compHardwareDtypeDict = {'Vendor Name': 'str','Model Name': 'str','MYCT': 'int','MMIN': 'int',
                             'MMAX': 'int','CACH': 'int','CHMIN': 'int','CHMAX': 'int','PRP': 'int','ERP': 'int'}
    
    #Create an Instance of the Data Set Class
    compHardwareDS = DataSetHelper.DataSet(compHardwareDataSetName,
                                           compHardwareDataSet_OverallType,
                                           compHardwareDataSet_Predictor,
                                           compHardwareDataSetPath,
                                           compHardwareHeader,
                                           compHardwareDtypeDict
                                           )
    
    #Add to the Overall Data Set Dictonary 
    allDataSetObjects[compHardwareDataSetName] = compHardwareDS
    
    ####################
    # ALBALONE DATA
    ####################
    abaloneDataSetName = 'Albalone'
    abaloneDataSet_OverallType = 'Regression'
    abaloneDatasSet_Predictor = 'Rings'
    abaloneDataSetPath = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\Abalone\abalone.data"
    abaloneHeaders = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
    abaloneDtypeDict = {'Sex': 'str', 'Length': 'float' , 'Diameter': 'float', 'Height': 'float', 
                        'Whole Weight':'float', 'Shucked Weight':'float', 'Viscera Weight': 'float', 'Shell Weight': 'float', 'Rings': 'int'}
    
    #Create an Instance of the Data Set Class
    albaloneDS = DataSetHelper.DataSet(abaloneDataSetName,
                                       abaloneDataSet_OverallType,
                                       abaloneDatasSet_Predictor,
                                       abaloneDataSetPath,
                                       abaloneHeaders,
                                       abaloneDtypeDict
                                       )
    
    #Add to the Overall Data Set Dictonary 
    allDataSetObjects[abaloneDataSetName] = albaloneDS
    
    
    ####################
    # FOREST FIRE DATA
    ####################
    forestFireDataSetName = 'Forest Fire'
    forestFireDataSet_OverallType = 'Regression'
    foresetFireDataSet_Predictor = 'area'
    forestFireDataSetPath = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\ForestFires\forestfires.data"
    
    #Create an Instance of the Data Set Class
    forestFireDS = DataSetHelper.DataSet(forestFireDataSetName,
                                         forestFireDataSet_OverallType,
                                         foresetFireDataSet_Predictor,
                                         forestFireDataSetPath
                                         )
    
    #Add to the Overall Data Set Dictonary 
    allDataSetObjects[forestFireDataSetName] = forestFireDS
    
    
    ####################
    # EXAMPLE DT DATA (Categorical)
    ####################
    #Day, Outlook, Temperature, Humidity, Wind, Play Tennis
    tennisDataSetName = 'Tennis'
    tennisDataSet_OverallType = 'Classification'
    tennisDataSet_Predictor = 'Play Tennis'
    tennisDataSetPath = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\DTreeExample.data"
    tennisID3Types = {'Day': 'Num','Outlook': 'Cat','Temperature': 'Cat','Humidity': 'Cat','Wind': 'Cat','Play Tennis': 'Cat'}
    
    expDeTreeDS = DataSetHelper.DataSet(tennisDataSetName,
                                         tennisDataSet_OverallType,
                                         tennisDataSet_Predictor,
                                         tennisDataSetPath,
                                         None,
                                         None,
                                         None,
                                         None,
                                         tennisID3Types)
    
    #Add to the Overall Data Set Dictonary 
    allDataSetObjects[tennisDataSetName] = expDeTreeDS
    
    ####################
    # EXAMPLE DT DATA (Numeric)
    ####################
    #Day, Temp, Humd, Wind, Play Tennis
    numTennisDataSetName = 'NumberTennis'
    numTennisDataSet_OverallType = 'Classification'
    numTennisDataSet_Predictor = 'Play Tennis'
    numTennisDataSetPath = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\DTreeExample_Numeric.data"
    numTennisID3Types = {'Day': 'Num','Temp': 'Num','Humd': 'Num','Play Tennis': 'Cat'}
    
    expDeTreeDS2 = DataSetHelper.DataSet(numTennisDataSetName,
                                         numTennisDataSet_OverallType,
                                         numTennisDataSet_Predictor,
                                         numTennisDataSetPath,
                                         None,
                                         None,
                                         None,
                                         None,
                                         numTennisID3Types)
    
    #Add to the Overall Data Set Dictonary 
    allDataSetObjects[numTennisDataSetName] = expDeTreeDS2
    
    return allDataSetObjects

if __name__ == "__main__":
    print("MLPL - Machine Learning Pipeline - ID3 and CART")
    #Load in the Data and Set up Basic Data Sets
    allDataSets = defineAllDataSets()
    
    #Set up the KCross Val Helper 
    myKCrossValHelper = KCrossValHelperModule.KCrossValHelper(allDataSets)
    
    #Set up the ID3 Algo Helper
#    id3_TennisHelper = ID3HelperModule.ID3Helper(allDataSets['Tennis'].name, 2, 'Play Tennis')
#    id3_TennisHelper.generateTree(allDataSets['Tennis'].finalData)
    
    
    #TODO: Need to wrap this up such that it happens for each parition 
#    entPar = id3_TennisHelper._calcPartitionEntropy(allDataSets['Tennis'].finalData)
#    expEnt = id3_TennisHelper._calcExpectedEntropyAllFeaturesInCurrentParition(allDataSets['Tennis'].finalData)
#    gainPar = id3_TennisHelper._calcGainAllFeaturesInCurrentParition(entPar, expEnt)
#    infoValPar = id3_TennisHelper._calcInformationValueAllFeaturesInCurretPartition(allDataSets['Tennis'].finalData)
#    gainRatio = id3_TennisHelper._calGainRatioAllFeaturesInCurrentPartition(gainPar, infoValPar)
#    maxGainRatio = max(gainRatio, key=gainRatio.get)
#    print(maxGainRatio)
    
    #https://machinewithdata.com/2018/07/10/how-to-calculate-gain-ratio/
    #https://www.cse.unsw.edu.au/~cs9417ml/DT1/decisiontreealgorithm.html
    #https://machinewithdata.com/2020/06/17/deriving-decision-tree-using-entropy-id3-approach/
    #https://beginningwithml.wordpress.com/2019/01/14/9-2-improving-the-id3-algorithm/
    #https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation#:~:text=Stratification%20is%20the%20process%20of,comprises%20around%20half%20the%20instances.
    
    #TODO: Then also need to pick a way that the one with the max gain raito is the base of the tree
    #This would be a parition that only has data from One Feature in it. (the feature with max gain)
    #Also need to figure how when to "stop" the recursive calls when building the tree
    
    
    #Set up the ID3 Algo Helper
#    id3_TennisHelper = ID3HelperModule.ID3Helper(allDataSets['Tennis'].name, 2, 'Play Tennis', 'Day', allDataSets)
#    tennisFinalID3Data = id3_TennisHelper.dropUniqueIDs(allDataSets['Tennis'].finalData)
#    id3_TennisHelper.runID3Algo(tennisFinalID3Data)
    
    #myKCrossValHelper.runKFoldCrossVal_ID3_Univariate('Tennis', 'Play Tennis', 2, 'Day')
    #myKCrossValHelper.runKFoldCrossVal_ID3_Univariate('NumberTennis', 'Play Tennis', 2, 'Day')
    #myKCrossValHelper.runKFoldCrossVal_ID3_Univariate('Breast Cancer', 'Class', 2, 'Sample Code Number')
    
    #myKCrossValHelper.runKFoldCrossVal_ID3_Univariate('Congressional Vote', 'Class Name', 2, None)
    
    myKCrossValHelper.runKFoldCrossVal_ID3_Univariate('Car Eval', 'Car Acceptability', 4, None)
    
    #id3_BreastCancerHelper = ID3HelperModule.ID3Helper(allDataSets['Breast Cancer'].name, 2, 'Class', 'Sample Code Number', allDataSets)
    
#    id3_Num_TennisHelper = ID3HelperModule.ID3Helper(allDataSets['NumberTennis'].name, 2, 'Play Tennis', 'Day', allDataSets)
#    numTennisFinalID3Data = id3_Num_TennisHelper.dropUniqueIDs(allDataSets['NumberTennis'].finalData)
#    id3_Num_TennisHelper.runID3Algo(numTennisFinalID3Data)
    
    
#    id3_BreastCancerHelper = ID3HelperModule.ID3Helper(allDataSets['Breast Cancer'].name, 2, 'Class', 'Sample Code Number', allDataSets)
#    bcFinalID3Data = id3_BreastCancerHelper.dropUniqueIDs(allDataSets['Breast Cancer'].finalData)
#    id3_BreastCancerHelper.runID3Algo(bcFinalID3Data)
    
    
    
    
    
    
    
    
    
    
    
    

