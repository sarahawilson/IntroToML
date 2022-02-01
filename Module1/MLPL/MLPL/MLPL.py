import DataHelper

if __name__ == "__main__":
    print("MLPL - Machine Learning Pipeline")
    abaloneDataSet = r"C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\DataSets\Abalone\abalone.data"
    myDH_abalone = DataHelper.DataHelper(abaloneDataSet)
    myDH_abalone.printData()
