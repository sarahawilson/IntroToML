# Sarah Wilson 
# 303 - 921 - 7225
# Project 2
# Introduction to Machine Learning

from typing import List, Tuple, Dict;
import DataSetHelper 
import KCrossValHelperModule


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
                             breastCancerApplyConversionAttributes
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
    
    #Create an Instance of the Data Set Class
    carEvalDS = DataSetHelper.DataSet(carEvalDataSetName,
                                      carEvalDataSet_OverallType,
                                      carEvalDataSet_Predictor,
                                      carEvalDataSetPath,
                                      carEvalHeaders,
                                      carDtypes,
                                      carMissingValAttributes,
                                      carApplyConversionValueAttribues
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
    
    #Create an Instance of the Data Set Class
    congVoteDS = DataSetHelper.DataSet(congVoteDataSetName,
                                       congVoteDataSet_OverallType,
                                       congVoteDataSet_Predictor,
                                       congVoteDataSetPath,
                                       congVoteHeaders)
    
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
    
    return allDataSetObjects

if __name__ == "__main__":
    print("MLPL - Machine Learning Pipeline")
    #Load in the Data and Set up Basic Data Sets
    allDataSets = defineAllDataSets()
    
    #Set up the KCross Val Helper 
    myKCrossValHelper = KCrossValHelperModule.KCrossValHelper(allDataSets)
    
    #Create the Tune and Expirment Sets for all the Data sets
    myKCrossValHelper.createValidation_TuneAndExperimentSets()
    
    
    #Run the Tuning on the Classification Tasks
    #myKCrossValHelper.runKFoldCrossVal_OnSingleDataSet_ForTuningKValKNN('Car Eval')
    #myKCrossValHelper.runKFoldCrossVal_OnSingleDataSet_ForTuningKValKNN('Breast Cancer')
    #myKCrossValHelper.runKFoldCrossVal_OnSingleDataSet_ForTuningKValKNN('Congressional Vote')
    
    #Classification Tasks 
    #Define the best K value in KNN to run the Full Expirment With
    #These numbers were determine by looking at the results from the runKFoldCross value on 
    #knn from the tuning process
    congVoteOptK = 3
    breastCancerOptK = 7
    carEvalOptKOptK = 3
    
    #Run the Expirment on the Classificaiton Task, with optimal k values
    #myKCrossValHelper.runKFoldCrossVal_OnSingleDataSet_ForExp('Car Eval', carEvalOptKOptK)
    #myKCrossValHelper.runKFoldCrossVal_OnSingleDataSet_ForExp('Breast Cancer', breastCancerOptK)
    #myKCrossValHelper.runKFoldCrossVal_OnSingleDataSet_ForExp('Congressional Vote', congVoteOptK)
    
    
    #Run the Tuning for K (kNN) and Sigma on the Regression Tasks
    #myKCrossValHelper.runKFoldCrossVal_OnSingleDataSet_ForTuningSigmaAndK('Albalone')
    albaloneOptK = 7
    albaloneOptSigma = 0.1
    
    myKCrossValHelper.runKFoldCrossVal_OnSingleDataSet_ForExp('Albalone', albaloneOptK, albaloneOptSigma)
    
    
    #myKCrossValHelper.runKFoldCrossVal_OnSingleDataSet_ForTuningSigmaAndK('Forest Fire')
    #myKCrossValHelper.runKFoldCrossVal_OnSingleDataSet_ForTuningSigmaAndK('Computer Hardware')
    
    #Regression Tasks Optimal K (Normal KNN)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

