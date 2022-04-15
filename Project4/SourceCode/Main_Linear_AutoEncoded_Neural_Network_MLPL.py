# Sarah Wilson 
# 303 - 921 - 7225
# Project 4
# Introduction to Machine Learning


##### MAIN FOR AUTOENCODED LINEAR NN

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
    
    #Final Clean Up
    #Convert all the data to Numeric 
    DataSetHelper.ConvertDataSetsToNumeric(allDataSetObjects)
    
    
    #For Linear and Logistic Regression Scale the Values to be between 
    # 0-1
    DataSetHelper.NormailzeDataSets(allDataSetObjects)

        
    return allDataSetObjects

if __name__ == "__main__":
    print("MLPL - Machine Learning Pipeline - Linear Neural Networks")
    #Load in the Data and Set up Basic Data Sets
    allDataSets = defineAllDataSets()
    
    
    #Set up the KCross Val Helper 
    myKCrossValHelper = KCrossValHelperModule.KCrossValHelper(allDataSets)
    
    
    ##TUNING!!!!!
    learningRateN = [0.001, 0.01, 0.1, 1]
    convergeFactorEP = [10, 20]
    
    #myKCrossValHelper.runKFoldCrossVal_Linear_AutoEncoded_NN_Tune('Breast Cancer', learningRateN, convergeFactorEP, 2, 2, 4)
    #myKCrossValHelper.runKFoldCrossVal_Linear_AutoEncoded_NN_Tune('Congressional Vote', learningRateN, convergeFactorEP, 2, 'republican', 'democrat')
    #myKCrossValHelper.runKFoldCrossVal_Linear_AutoEncoded_NN_Tune('Computer Hardware', learningRateN, convergeFactorEP, None, None, None)
    myKCrossValHelper.runKFoldCrossVal_Linear_AutoEncoded_NN_Tune('Forest Fire', learningRateN, convergeFactorEP, None, None, None)
    #myKCrossValHelper.runKFoldCrossVal_Linear_AutoEncoded_NN_Tune('Albalone', learningRateN, convergeFactorEP, None, None, None)
    
    #    #Optimal Values Gathered from the Tuning
    compHW_N = 0.001 
    compHW_EP = 10
#    
    albalone_N = 0.01
    albalone_EP = 100
#    
    forestFire_N = 1
    forestFie_EP = 10
#    
    congVote_N = 1
    congVote_EP = 10
#    
    bc_N = 0.1
    bc_EP = 100
#    
#    carEval_N = 0
#    carEval_EP = 0
    
    #myKCrossValHelper.runKFoldCrossVal_Linear_AutoEncoded_NN('Breast Cancer', 1, 10, 2, 2, 4)
    #myKCrossValHelper.runKFoldCrossVal_Linear_AutoEncoded_NN('Congressional Vote', congVote_N, congVote_EP, 2, 'republican', 'democrat')
    myKCrossValHelper.runKFoldCrossVal_Linear_AutoEncoded_NN('Computer Hardware', compHW_N, compHW_EP, None, None, None)
    #myKCrossValHelper.runKFoldCrossVal_Linear_AutoEncoded_NN('Forest Fire', forestFire_N, forestFie_EP, None, None, None)
    #myKCrossValHelper.runKFoldCrossVal_Linear_NN('Congressional Vote', congVote_N, congVote_EP, 2, 'republican', 'democrat')
    #myKCrossValHelper.runKFoldCrossVal_Linear_AutoEncoded_NN('Albalone', albalone_N, albalone_EP, None, None, None)
    

    
    

    
    

    
    
    
    
    
    
    
    
    
    
    
    

