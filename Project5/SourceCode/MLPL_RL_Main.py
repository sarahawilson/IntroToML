# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

import ImportHelperModule
import ValueIterationModule
import QLearningModule
import SARSAModule

# User Defined Varaibles 
trackName = 'O'

#Defines Which Algorithm to Use
# V = Value Iteration
# Q = Q Values
# S = SARSA
algorithm = 'S'

#Define the Crash Logic 
#True = Car goes back to start position with zero'd velocity
harshCrash = False

#Define if we want the results plotter on 
plotter_V = True
plotter_Q = True
plotter_S = True

 

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
    avgTimeTaken = []

    valIterHelper = ValueIterationModule.ValIterHelper(curRaceTrack, harshCrash)
    
    #Run the Value Iteration Based on the Best 
    for epsilonVal in tuneEpsilonList:
        for discountVal in tuneDiscountList:
            metricForLearningPlot = valIterHelper.runTrain(epsilonVal, numIterations, discountVal)
            if (plotter_V):
                valIterHelper.plotLearningCurve(epsilonVal, metricForLearningPlot, trackName)
            
            #Get a Statistical Model for how well the training went 
            # Run 10 Verison of the Car
            for runNum in range(10):
                timeTaken = valIterHelper.runTest()
                print('Time for Current Run:')
                print(str(timeTaken))
                avgTimeTaken.append(timeTaken)
            #Calculate the Average time over those ten runs
            avgSum = 0
            for time in avgTimeTaken:
                avgSum = avgSum + time
            avgTime = avgSum/10
            print('Average Time Across 10 Runs:')
            print(str(avgTime))
            #curMetric = (epsilonVal, discountVal, timeTaken)
            #metricForRuns.append(curMetric)
    valIterHelper.dumpResultsToFile(trackName)
            
elif(algorithm == 'Q'):
    #tuneDiscountList = [.1,.5,.7]
    tuneDiscountList = [.90]
    #tuneEpsilonList = [0.01,0.1,1]
    tuneEpsilonList = [0.20]
    tuneLearnRate = [0.60]
    numTrain_Iterations = 10000
    numMoves_Allowed = 1000
    avgTimeTaken = []
    
    qLearnHelper = QLearningModule.QLearnHelper(curRaceTrack, harshCrash)
    #Run the Q Learning Over the Tuning Parameters
    for epsilonVal in tuneEpsilonList:
        for discountVal in tuneDiscountList:
            for learnVal in tuneLearnRate:
                metricForLearningPlot = qLearnHelper.runTrain(epsilonVal, learnVal, discountVal, numTrain_Iterations, numMoves_Allowed)
                if (plotter_Q):
                    qLearnHelper.plotLearningCurve(metricForLearningPlot, trackName)
                #print(qLearnHelper.q_table[(1,1), (0,0)])
                #Get a Statistical Model for how well the training went 
                # Run 10 Verison of the Car
                for runNum in range(10):
                    timeTaken = qLearnHelper.runTest(0, 1000)
                    print('Time for Current Run:')
                    print(str(timeTaken))
                    avgTimeTaken.append(timeTaken)
                #Calculate the Average time over those ten runs
                avgSum = 0
                for time in avgTimeTaken:
                    avgSum = avgSum + time
                avgTime = avgSum/10
                print('Average Time Across 10 Runs:')
                print(str(avgTime))
    qLearnHelper.dumpResultsToFile(trackName)           

    
elif(algorithm == 'S'):
    #tuneDiscountList = [.1,.5,.7]
    tuneDiscountList = [.90]
    #tuneEpsilonList = [0.01,0.1,1]
    tuneEpsilonList = [0.20]
    tuneLearnRate = [0.60]
    numTrain_Iterations = 10000
    numMoves_Allowed = 1000
    metricForRuns = []
    
    sarsaHelper = SARSAModule.SARSAHelper(curRaceTrack, harshCrash)
    #Run the Q Learning Over the Tuning Parameters
    for epsilonVal in tuneEpsilonList:
        for discountVal in tuneDiscountList:
            for learnVal in tuneLearnRate:
                metricForLearningPlot = sarsaHelper.runTrain(epsilonVal, learnVal, discountVal, numTrain_Iterations, numMoves_Allowed)
                if (plotter_S):
                    sarsaHelper.plotLearningCurve(metricForLearningPlot, trackName)
                    
                #Get a Statistical Model for how well the training went 
                # Run 10 Verison of the Car
                for runNum in range(10):
                    timeTaken = sarsaHelper.runTest(0, 1000)
                    print('Time for Current Run:')
                    print(str(timeTaken))
                    avgTimeTaken.append(timeTaken)
                #Calculate the Average time over those ten runs
                avgSum = 0
                for time in avgTimeTaken:
                    avgSum = avgSum + time
                avgTime = avgSum/10
                print('Average Time Across 10 Runs:')
                print(str(avgTime))   
    sarsaHelper.dumpResultsToFile(trackName)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
