# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

from typing import Tuple, Dict
from itertools import product
import copy
import numpy as np
import random
import matplotlib.pyplot as plt

import RaceTrackModule
import CarModule

class QLearnHelper:
    #Helper module the run the Q-Learning ALgorithm
    def __init__(self, RaceTrack, harshCrashLogic):
        self.raceTrack = RaceTrack
        self.actionSpace = ActionSpace_A()
        self.q_table = self._init_q_table()
        self.harshCrashLogic = harshCrashLogic
        self.fileContentsToWrite = []

    def _init_q_table(self):
        #Q table has the same Shape as stateSpace
        # but has the added dim of action
        
        #Acceleartion falls between [-1,1]
        range_acc_x = [-1,0,1]
        range_acc_y = [-1,0,1]
        accCombos =  list(product(range_acc_x, range_acc_y))
        
        #Velocity falls between [-5, 5]
        range_vel_x = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        range_vel_y = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        velCombos =  list(product(range_vel_x, range_vel_y))  
        
        qTable_of_SA = {}
        qTable_of_A = {}
        # Build the loop up to get Q(s)
        # Use a nested dictonary 
        numPos = 0
        for position in self.raceTrack.raceTrackLayout:
            numPos = numPos + 1
            curEnd = False
            for endPosition in self.raceTrack.endPositions:
                if (position == endPosition):
                    curEnd = True
                    break
                
            for velocity in velCombos:
                pos_vel_tupleKey = (position, velocity)

                #Now build the q table related to accerelation
                qTable_of_A = {}
                for acceleration in accCombos:
                    if(curEnd):
                        qTable_of_A[acceleration] = 0
                    else:
                        qTable_of_A[acceleration] = -1000
                #append to overall q table

                qTable_of_SA[pos_vel_tupleKey] = qTable_of_A
        return qTable_of_SA
    
    def _greedy_epsilon(self, cur_q_a_valueInS, epsilon):
        # Implements the greedy epsilon algorithm
        # for if exploration or execution will be used for the next 
        # action
        
        ep_acceleration = None
        #Generate a random number (like a dice roll)
        diceRoll = np.random.random()
        if (diceRoll < epsilon):
            # Randomly pick acceleration x and y
            acc_x = random.randint(-1,1)
            acc_y = random.randint(-1,1)
            ep_acceleration = (acc_x, acc_y)
        else:
            # Pick the best action based on the argMax of the current q values
            maxAccessKey = max(cur_q_a_valueInS, key=cur_q_a_valueInS.get)
            ep_acceleration = maxAccessKey
        
        return ep_acceleration
    
    
    def runTrain(self, epsilon, learningRate, discount, iterations, movesAllowed):
       # Runs the training on the car for Q-Learning 
        metricsForPlots = []
        for iteration in range(iterations):
            # Set up the Car
            # In Q-Learning the Car Explores the Space to determine the policy
            curCar = CarModule.Car(self.harshCrashLogic, self.raceTrack.raceTrackLayout, self.raceTrack.width, self.raceTrack.height)
            #curCar.init_car_kinematics(self.raceTrack.startPosition)
            curCar.init_car_rando(self.raceTrack.startPosition)
            
            zz_curPos = curCar.curPosition
            zz_curVel = curCar.curVelocity
    
                        
            # Training Process runs until the car reaches the finish line
            # or number of iterations has been reached
            reachedFinishLine = False
            loopIterations = 0
            while ((not reachedFinishLine) and (loopIterations < movesAllowed)):
                zz_curPos = curCar.curPosition
                zz_curVel = curCar.curVelocity
                
                q_s_AccessKey = (curCar.curPosition, curCar.curVelocity)
                cur_q_a_values_basedOnCarState = self.q_table[q_s_AccessKey]
                
                #Let the greedy epsilon algorithm pick the acceleration to use
                ep_picked_acceleration = self._greedy_epsilon(cur_q_a_values_basedOnCarState, epsilon)
                
                # Non Deterministic Car Motion
                # Roll dice 
                if (random.random() < 0.8):
                    # Acceleration is applied
                    q_s_a_value =  cur_q_a_values_basedOnCarState[ep_picked_acceleration]
                    accToApply = ep_picked_acceleration
                else:
                    accToApply = (0,0)
                    q_s_a_value = cur_q_a_values_basedOnCarState[accToApply]
                   
                    
                zza_accToApply = accToApply
                # Apply Acceleration to Car
                #Apply Acceleariton First (changes velocity)
                curCar.applyAcceleartion(accToApply)
                #Then Apply the Velocity (changes position) 
                reachedFinishLine = curCar.applyVelocity()
                
                zzn_curPos = curCar.curPosition
                zzn_curVel = curCar.curVelocity
                
                if (not reachedFinishLine):
                    newPos = curCar.curPosition
                    newVel = curCar.curVelocity
                    
                    # Get Q(s')
                    q_s_primeAccessKey = (newPos, newVel)
                    q_s_prime = self.q_table[q_s_primeAccessKey]
                    
                    # Determine Max a' based on Q(s')
                    max_q_acc_prime_AccessKey = max(q_s_prime, key=q_s_prime.get)
                    max_q_acc_prime = q_s_prime[max_q_acc_prime_AccessKey]
                    
                    # Update the q_s_a values
                    reward = -1
                    cur_q_a_values_basedOnCarState[accToApply] += (learningRate *(reward + (discount * max_q_acc_prime) - q_s_a_value))
                
                loopIterations = loopIterations + 1
                
            epsilon = epsilon * 0.99
                
            metricsForPlots.append(loopIterations)
        
        print('--Training the Race Car--')
        self.fileContentsToWrite.append('--Training the Race Car--')
        if(self.harshCrashLogic):
            print('Bad Crash')
            self.fileContentsToWrite.append('Bad Crash')
        else:
            print('Simple Crash')
            self.fileContentsToWrite.append('Simple Crash')
        print('Training Iterations:' + str(iterations))
        self.fileContentsToWrite.append('Training Iterations:' + str(iterations))
        
        movesSum = 0
        for metric in metricsForPlots:
            movesSum = movesSum + metric
        moveAvg = (movesSum / len(metricsForPlots))
        print('Avg Number of Steps Taken:'  + str(moveAvg))
        self.fileContentsToWrite.append('Avg Number of Steps Taken:'  + str(moveAvg))
        print('-------------------------')
        self.fileContentsToWrite.append('-------------------------')
        
        
        return metricsForPlots
            
            
                
    def runTest(self, epsilon, iterations):
        # Runs the Test on the Car in the Race Track based on the results learned in the training
         
        #Set up the Car 
        curCar = CarModule.Car(self.harshCrashLogic, self.raceTrack.raceTrackLayout, self.raceTrack.width, self.raceTrack.height)
        curCar.init_car_kinematics(self.raceTrack.startPosition)
        print('--Testing the Race Car--')
        self.fileContentsToWrite.append('--Testing the Race Car--')
        if(self.harshCrashLogic):
            print('Bad Crash')
            self.fileContentsToWrite.append('Bad Crash')
        else:
            print('Simple Crash')
            self.fileContentsToWrite.append('Simple Crash')
        print('Car Starting Position:')
        self.fileContentsToWrite.append('Car Starting Position:')
        print('Car Position' + str(curCar.curPosition))
        self.fileContentsToWrite.append('Car Position' + str(curCar.curPosition))
        print('Car Velocity' + str(curCar.curVelocity))
        self.fileContentsToWrite.append('Car Velocity' + str(curCar.curVelocity))
        
        # Let the car run around the track until it has reached the finish line
        reachedFinishLine = False
        
        elapsedTime = 0 
        loopIterations = 0
        while ((not reachedFinishLine) and (loopIterations < iterations)):
            q_s_AccessKey = (curCar.curPosition, curCar.curVelocity)
            cur_q_a_values_basedOnCarState = self.q_table[q_s_AccessKey]
            
            #Let the greedy epsilon algorithm pick the acceleration to use
            maxAccessKey = max(cur_q_a_values_basedOnCarState, key=cur_q_a_values_basedOnCarState.get)
            #ep_picked_acceleration = self._greedy_epsilon(cur_q_a_values_basedOnCarState, epsilon)
            bestacc = maxAccessKey
            
            print('Car Acceleration:' + str(bestacc))
            self.fileContentsToWrite.append('Car Acceleration:' + str(bestacc))
            
            # Apply Acceleration to Car
            #Apply Acceleariton First (changes velocity)
            curCar.applyAcceleartion(bestacc)
            print('Car Next State:')
            self.fileContentsToWrite.append('Car Next State:')
            print('Car Position' + str(curCar.curPosition))
            self.fileContentsToWrite.append('Car Position' + str(curCar.curPosition))
            print('Car Velocity' + str(curCar.curVelocity))
            self.fileContentsToWrite.append('Car Velocity' + str(curCar.curVelocity))
            
            elapsedTime = elapsedTime + 1
            loopIterations = loopIterations + 1
            #Then Apply the Velocity (changes position) 
            reachedFinishLine = curCar.applyVelocity()
            
        return elapsedTime    
    
    
    
    def plotLearningCurve(self, metricsList, trackName):
        # Plots the Learning Curve for the Value Iteration Algorithm
        iterations = []
        iterNum = 1
        for metric in metricsList:
            iterations.append(iterNum)
            iterNum = iterNum + 1
            
        plt.plot(iterations, metricsList)
        #plt.axhline(y=epsilon, color='r', linestyle='-')
        
        plt.xlabel('Number of Iterations')
        plt.ylabel('Number of Moves Car Took')
        
        if(trackName == 'T'):
            nameTrack = 'T-Track'
        elif(trackName == 'R'):
            nameTrack = 'R-Track'
        elif(trackName == 'O'):
            nameTrack = 'O-Track'
        elif(trackName == 'L'):
            nameTrack = 'L-Track'
        
        if(self.harshCrashLogic):
            crashType = ' Crash Type 2 (Start)'
        else:
            crashType = ' Crash Type 1 (Nearest)'
        
        plotTitle = 'Q_Learning: ' + nameTrack + crashType
        plt.title(plotTitle)
        
        plt.show()
        
    def dumpResultsToFile(self, trackName):
        # writes out run results to a .txt file
        filePath = r'C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\Project5\WilsonProject#5\SourceOutput'
        if(trackName == 'T'):
            nameTrack = '_T-Track'
        elif(trackName == 'R'):
            nameTrack = '_R-Track'
        elif(trackName == 'O'):
            nameTrack = '_O-Track'
        elif(trackName == 'L'):
            nameTrack = '_L-Track'
        
        if(self.harshCrashLogic):
            crashType = '_Crash_2_Start'
        else:
            crashType = '_Crash_1_Nearest'
        
        fileName = '\QLearn' + crashType + nameTrack + '.txt'
        
        fullPath = filePath + fileName
        textfile = open(fullPath, "w")
        for item in self.fileContentsToWrite:
            textfile. write(item + "\n")
        textfile. close()
        
        
class ActionSpace_A:
    #Builds teh action space A based on all the possible accelerations
    def __init__(self):
        self.actions_A = self.defineActionSpace()
        self.propApplied = 0.8
        self.propNotApplied = 0.2
        
    def defineActionSpace(self):
        # Defines the set of all possible actions
        # Tuple (acc_x, acc_y)
        # There should be 9 unique tuples of acc_x and acc_y
        # acc_x and acc_y can be -1, 0, 1
        range_acc_x = [-1,0,1]
        range_acc_y = [-1,0,1]
        accCombos =  list(product(range_acc_x, range_acc_y))
        return accCombos
    

