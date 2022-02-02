#Sarah Wilson 


import DataHelper

if __name__ == "__main__":
    print("MLPL - Machine Learning Pipeline")
    
    #Data File Paths
    abaloneDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\Abalone\abalone.data"
    carEvalDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\CarEvaluation\car.data"
    breastCancerDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\BreastCancer\breast-cancer-wisconsin.data"
    compHardwareDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\ComputerHardware\machine.data"
    congVoteDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\CongressionalVote\house-votes-84.data"
    forestFireDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\ForestFires\forestfires.data"
    
    testAvgDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\simpleTestDataSets\colAvg.data"
    
    dsLocations = [abaloneDataSet, carEvalDataSet, breastCancerDataSet, compHardwareDataSet, congVoteDataSet, forestFireDataSet]
    
    #Albalone Data Set NH 
    #Car Data Set has NMV NH
    #Forest Fires has NMV Header
    #
    
    myDH_Abalone = DataHelper.DataHelper(abaloneDataSet, False)
    myDH_CarEval = DataHelper.DataHelper(carEvalDataSet, False)
    myDH_BreastCancer = DataHelper.DataHelper(breastCancerDataSet, False)
    myDH_CompHW = DataHelper.DataHelper(compHardwareDataSet, True)
    myDH_CongVote = DataHelper.DataHelper(congVoteDataSet, False)
    myDH_ForestFire = DataHelper.DataHelper(forestFireDataSet, True)
    
    myDH_TestAvg = DataHelper.DataHelper(testAvgDataSet, False)
    
    #Handle the Missing Data
    myDH_Abalone.handleMissingValues() #No Missing Data
    myDH_CarEval.handleMissingValues() #No Missing Data
    #myDH_BreastCancer.handleMissingValues('?') #Missing Value represented by ?
    myDH_CompHW.handleMissingValues() #No Missing Data
    myDH_CongVote.handleMissingValues() #No Missing Data (? has different meaning)
    myDH_ForestFire.handleMissingValues() #No Missing Data 
    
    
    myDH_TestAvg.printData()
    myDH_TestAvg.handleMissingValues("?")
    myDH_TestAvg.printData()