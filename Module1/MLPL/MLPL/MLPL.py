#Sarah Wilson 


import DataHelper
import pandas as pd

if __name__ == "__main__":
    print("MLPL - Machine Learning Pipeline")
    
    #Data File Paths
    abaloneDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\Abalone\abalone.data"
    abaloneHeaders = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
    #abaloneDtypeDict = {'Sex': 'str', 'Length': , 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings'}
    #https://www.roelpeters.be/solved-dtypewarning-columns-have-mixed-types-specify-dtype-option-on-import-or-set-low-memory-in-pandas/
    #Basically use converters to get each column to read in as the right type
    
    
    carEvalDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\CarEvaluation\car.data"
    carEvalHeaders = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_Boot', 'Safety']
    
    breastCancerDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\BreastCancer\breast-cancer-wisconsin.data"
    breastCancerHeaders = ['Sample Code Number', 'Clump Thickness', 'Uni. of Cell Size', 'Uni. of Cell Shape', 'Marginal Adhesion', 'Single Ep. Cell Size', 'Bare Nuclei',
                            'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    
    compHardwareDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\ComputerHardware\machine.data"
    compHardwareHeader = ['Vendor Name','Model Name','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
    
    congVoteDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\CongressionalVote\house-votes-84.data"
    congVoteHeaders = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                       'religious-groups-in-schools','anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback'
                       'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
    
    forestFireDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\ForestFires\forestfires.data"
    
    testAvgDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\colAvg.data"
    testAvgDataHeaders = ['Col0','Col1','Col2']
    df_testAvgDataSet = pd.read_csv(testAvgDataSet, names=testAvgDataHeaders)
        
    df_Abalone = pd.read_csv(abaloneDataSet, names=abaloneHeaders)
    df_CarEval = pd.read_csv(carEvalDataSet, names=carEvalHeaders)
    df_BreastCancer = pd.read_csv(breastCancerDataSet, names=breastCancerHeaders)
    df_CompHW = pd.read_csv(compHardwareDataSet, names=compHardwareHeader)
    df_CongVote = pd.read_csv(congVoteDataSet, names=congVoteHeaders)
    df_ForestFire = pd.read_csv(forestFireDataSet)


    myDH = DataHelper.DataHelper()
    print(df_testAvgDataSet)
    myDH.handleMissingValues(df_testAvgDataSet, '?')
    print(df_testAvgDataSet)
    
    
    #https://www.roelpeters.be/solved-dtypewarning-columns-have-mixed-types-specify-dtype-option-on-import-or-set-low-memory-in-pandas/
    
#    
#    myDH_TestAvg = DataHelper.DataHelper(testAvgDataSet, False)
#    
#    #Handle the Missing Data
#    myDH_Abalone.handleMissingValues() #No Missing Data
#    myDH_CarEval.handleMissingValues() #No Missing Data
#    #myDH_BreastCancer.handleMissingValues('?') #Missing Value represented by ?
#    myDH_CompHW.handleMissingValues() #No Missing Data
#    myDH_CongVote.handleMissingValues() #No Missing Data (? has different meaning)
#    myDH_ForestFire.handleMissingValues() #No Missing Data 
#    
#    #
#    myDH_TestAvg.printData()
#    myDH_TestAvg.handleMissingValues("?")
#    myDH_TestAvg.printData()