# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

import ImportHelperModule
import ValueIterationModule

# User Defined Varaibles 
trackName = 'O'

#Defines Which Algorithm to Use
# V = Value Iteration
# Q = Q Values
# S = SARSA
algorithm = 'V'

#Define the Crash Logic 
#True = Car goes back to start position with zero'd velocity
harshCrash = True 

 

# Init based on the user defined varaibles 
if(trackName == 'L'):
    raceTrackFile = r'C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\Project5\SourceCode\Tracks\L-track.txt'
elif(trackName == 'O'):
    raceTrackFile = r'C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\Project5\SourceCode\Tracks\O-track.txt'
elif(trackName == 'R'):
    raceTrackFile = r'C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\Project5\SourceCode\Tracks\R-track.txt'
elif(trackName == 'T'):
    raceTrackFile = r'C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\Project5\SourceCode\Tracks\T-track.txt'

#Load and Build the Track
impHelper = ImportHelperModule.ImportHelper(raceTrackFile)
curRaceTrack = impHelper.LoadRaceTrack()

# Run Value Iteration
if(algorithm == 'V'):
    #tuneDiscountList = [.1,.5,.7]
    tuneDiscountList = [.7]
    #tuneEpsilonList = [0.01,0.1,1]
    tuneEpsilonList = [0.000001]
    numIterations = 10000
    metricForRuns = []

    valIterHelper = ValueIterationModule.ValIterHelper(curRaceTrack, harshCrash)
    
    #Run the Value Iteration Based on the Best 
    for epsilonVal in tuneEpsilonList:
        for discountVal in tuneDiscountList:
            valIterHelper.runTrain(epsilonVal, numIterations, discountVal)
            timeTaken = valIterHelper.runTest()
            print(timeTaken)
            curMetric = (epsilonVal, discountVal, timeTaken)
            metricForRuns.append(curMetric)
            
elif(algorithm == 'Q'):
    #tuneDiscountList = [.1,.5,.7]
    tuneDiscountList = [.7]
    #tuneEpsilonList = [0.01,0.1,1]
    tuneEpsilonList = [0.000001]
    numIterations = 10000
    metricForRuns = []
