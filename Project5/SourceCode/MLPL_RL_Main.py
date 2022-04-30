# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

import ImportHelperModule
import ValueIterationHelperModule

# User Defined Varaibles 
trackName = 'T'

#Defines Which Algorithm to Use
# V = Value Iteration
# Q = Q Values
# S = SARSA
algorithm = 'V'

 

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
    tuneDiscountList = [.1,.5,.7]
    tuneEpsilonList = [0.01,0.1,1]
    timeToRunPerDiscount = []
    valIterHelper = ValueIterationHelperModule.ValIterHelper(curRaceTrack)
    valIterHelper.setInitialConditions()
    
    #Run the Value Iteration Based on the Best 
    for epsilonVal in tuneEpsilonList:
        for discountVal in tuneDiscountList:
            timeForRun = valIterHelper.run(discountVal, epsilonVal)
            timeToRunPerDiscount.append(timeForRun)
            valIterHelper.clearAllAndResetInitialConditions()
