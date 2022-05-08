# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

from typing import Tuple, Dict
from itertools import product
import copy
import random
import matplotlib.pyplot as plt

import RaceTrackModule
import CarModule


class ValIterHelper:
    # Helper Module to run the Value Iteration Algorihtm
    def __init__(self, RaceTrack, harshCrashLogic):
        self.raceTrack = RaceTrack
        self.actionSpace = ActionSpace_A()
        self.stateSpace = StateSpace_S(self.raceTrack.raceTrackLayout)
        self.q_table = self._init_q_table()
        self.value_table = self._init_value_table()
        self.policy_table = self._init_policy_table()
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
        
        qTable = {}
        for position in self.raceTrack.raceTrackLayout:
            for velocity in velCombos:
                for acceleration in accCombos:
                    pos_vel_acc_tupleKey = (position, velocity, acceleration)
                qTable[pos_vel_acc_tupleKey] = 0
        return qTable
        
    
    def _init_value_table(self):
        #Value Table will have the same Shape as stateSpace
        
        #Velocity falls between [-5, 5]
        range_vel_x = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        range_vel_y = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        velCombos =  list(product(range_vel_x, range_vel_y))      
        valueTable = {}
        for position in self.raceTrack.raceTrackLayout:
            for velocity in velCombos:
                pos_vel_tupleKey = (position, velocity)
                valueTable[pos_vel_tupleKey] = 0
        return valueTable
        
        
    def _init_policy_table(self):
        #Policy Table will have the same Shape as stateSpace
        
        #Velocity falls between [-5, 5]
        range_vel_x = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        range_vel_y = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        velCombos =  list(product(range_vel_x, range_vel_y))      
        policyTable = {}
        for position in self.raceTrack.raceTrackLayout:
            for velocity in velCombos:
                pos_vel_tupleKey = (position, velocity)
                policyTable[pos_vel_tupleKey] = 0
        return policyTable
    
    
    def plotLearningCurve(self, epsilon, metricsList, trackName):
        # Plots the Learning Curve for the Value Iteration Algorithm
        iterations = []
        iterNum = 1
        for metric in metricsList:
            iterations.append(iterNum)
            iterNum = iterNum + 1
            
        plt.plot(iterations, metricsList)
        #plt.axhline(y=epsilon, color='r', linestyle='-')
        
        plt.xlabel('Number of Iterations')
        plt.ylabel('Delta Max Value')
        
        if(trackName == 'T'):
            nameTrack = 'T-Track'
        elif(trackName == 'R'):
            nameTrack = 'R-Track'
        elif(trackName == 'O'):
            nameTrack = 'O-Track'
        elif(trackName == 'L'):
            nameTrack = 'L-Track'
        
        if(self.harshCrashLogic):
            crashType = 'Crash Type 2 (Start)'
        else:
            crashType = 'Crash Type 1 (Nearest)'
        
        plotTitle = 'Value Iteration: ' + nameTrack + crashType
        plt.title(plotTitle)
        
        plt.show()
        
    def dumpResultsToFile(self, trackName):
        #Write the run results to a .txt file
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
        
        fileName = '\ValIter' + crashType + nameTrack + '.txt'
        
        fullPath = filePath + fileName
        textfile = open(fullPath, "w")
        for item in self.fileContentsToWrite:
            textfile. write(item + "\n")
        textfile. close()
        
        
    def runTrain(self, epsilon, iterations, discount):
        # Runs the training for the value iteration algorithm
        curCar = CarModule.Car(self.harshCrashLogic, self.raceTrack.raceTrackLayout, self.raceTrack.width, self.raceTrack.height)
        curCar.init_car_kinematics(self.raceTrack.startPosition)
        
        metricForLearningPlot = []
        
        loopIterations = 0
        reachedConvergance = False
        while ((not reachedConvergance) and (loopIterations < iterations)):
            
            prev_ValueTable = copy.deepcopy(self.value_table)
            overall_delta_v = 0
            
            #For all s in S
            for pos_vel_state in self.stateSpace.states_S:
                #For all a in A
                max_q_val = -5
                policy_pi = (0,0) # Policy here is the acceeleration (acc_x, acc_y)
                
                for acceleariton_action in self.actionSpace.actions_A:
                    cur_pos_s = pos_vel_state[0]
                    cur_vel_s = pos_vel_state[1]
                    cur_acc_a = acceleariton_action
                    #Let the Car have the current state (position / velocity)
                    curCar.curPosition = cur_pos_s
                    curCar.curVelocity = cur_vel_s
                    
                    #Apply Acceleariton First (changes velocity)
                    curCar.applyAcceleartion(cur_acc_a)
                    #Then Apply the Velocity (changes position) 
                    reachedFinishLine = curCar.applyVelocity()
                    
                    
                    # Get the s_prime 
                    # When the acceleration asked for gets applied
                    pos_s_prime = curCar.curPosition
                    vel_s_prime = curCar.curVelocity
                    s_prime_AccessKey_accApplied = (pos_s_prime, vel_s_prime)
                    
                    # Get the s_prime
                    # When the accelration is NOT applied (acc_x =0, acc_y =0)
                    acc_a_none = (0,0)
                    #Set the Car back 
                    curCar.curPosition = cur_pos_s
                    curCar.curVelocity = cur_vel_s
                    curCar.applyAcceleartion(acc_a_none)
                    curCar.applyVelocity()
                    pos_s_prime_noAcc = curCar.curPosition
                    vel_s_prime_noAcc = curCar.curVelocity
                    
                    s_prime_AccessKey_accNotApplied = (pos_s_prime_noAcc, vel_s_prime_noAcc)
                    

                    #Calculate the Reward
                    if(reachedFinishLine):
                        reward = 0
                    else:
                        reward = -1
                        
                    value_s_prime_accApplied = prev_ValueTable[s_prime_AccessKey_accApplied]
                    value_s_prime_accNOTApplied = prev_ValueTable[s_prime_AccessKey_accNotApplied]
                    
                    # Calcualte the Probability or the Expected Value
                    curProb = (self.actionSpace.propApplied * value_s_prime_accApplied) + (self.actionSpace.propNotApplied * value_s_prime_accNOTApplied)

                    #Calcualte the new Q_t
                    Q_t = reward + discount*(curProb)
                    
                    #Update the Q table entires 
                    accessPosition = pos_vel_state[0]
                    accessVelocity = pos_vel_state[1]
                    accessAcceleration = acceleariton_action
                    qtableAccessKey = (accessPosition, accessVelocity,accessAcceleration)
                    
                    self.q_table[qtableAccessKey] = Q_t
                    
                    #Update the Policy (pi) and the maximum Q
                    if (Q_t > max_q_val):
                        policy_pi = acceleariton_action
                        max_q_val = Q_t
                    
                #Update the Tables to show the best action and policy
                prev_v_val = self.value_table[pos_vel_state]
                self.value_table[pos_vel_state] = max_q_val
                self.policy_table[pos_vel_state] = policy_pi
                    
                # Calcualte the difference bewtween the current
                # and previous v values
                delta_v_val = prev_v_val - max_q_val
                if(delta_v_val > overall_delta_v):
                    overall_delta_v = delta_v_val
                        
            #Check the Convergance Status
            if (overall_delta_v <= epsilon):
                reachedConvergance = True
            
            
            loopIterations = loopIterations + 1
            metricForLearningPlot.append(overall_delta_v)
        
        print('--Training the Race Car--')
        self.fileContentsToWrite.append('--Training the Race Car--')
        if(self.harshCrashLogic):
            print('Bad Crash')
            self.fileContentsToWrite.append('Bad Crash')
        else:
            print('Simple Crash')
            self.fileContentsToWrite.append('Simple Crash')
        print('Reached Convergance')
        self.fileContentsToWrite.append('Reached Convergance')
        print('Asked for Iterations:' + str(iterations))
        self.fileContentsToWrite.append('Asked for Iterations:' + str(iterations))
        print('Ran Iterations:'  + str(loopIterations))
        self.fileContentsToWrite.append('Ran Iterations:'  + str(loopIterations))
        print('Reached Convergance by Epsilon:' +str(reachedConvergance))
        self.fileContentsToWrite.append('Reached Convergance by Epsilon:' +str(reachedConvergance))
        print('-------------------------')
        self.fileContentsToWrite.append('-------------------------')
        
        return metricForLearningPlot
        
        
    def runTest(self):
        # Runs the Test on the Car using the policy that was obtained during the training
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
        
        
        reachedFinish = False
        elapsedTime = 0 
        while (not reachedFinish):
            #Get Car's Current Position
            curPos = curCar.curPosition
            curVel = curCar.curVelocity
            
            # 80% of the Time the Acceleration is Applied
            if (random.random() < 0.8):
                # Apply the intented Accelration
                #Use position and velocity key to look up the policy given it's state
                policyTableAccessKey = (curPos, curVel)
                bestPolicyAction = self.policy_table[policyTableAccessKey]
            else:
                # 20% of the Time NO Acceleration is Applied
                bestPolicyAction = (0,0)
            print('Car Acceleration:' + str(bestPolicyAction))
            self.fileContentsToWrite.append('Car Acceleration:' + str(bestPolicyAction))
            
            #Have the car perfrom the aciton given the policy
            curCar.applyAcceleartion(bestPolicyAction)
            print('Car Next State:')
            self.fileContentsToWrite.append('Car Next State:')
            print('Car Position' + str(curCar.curPosition))
            self.fileContentsToWrite.append('Car Position' + str(curCar.curPosition))
            print('Car Velocity' + str(curCar.curVelocity))
            self.fileContentsToWrite.append('Car Velocity' + str(curCar.curVelocity))
            
            elapsedTime = elapsedTime + 1
            reachedFinish = curCar.applyVelocity()
        
        return elapsedTime
            
            
            
        
class ActionSpace_A:
    #Build the aciton spaced based on the accelerations
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
        
        
class StateSpace_S:
    #Builds the state space based on how many possible poisition and veloicty 
    #combinations there are 
    def __init__ (self, raceTrackLayout):
        self.states_S = self.defineStateSpace(raceTrackLayout)

    def defineStateSpace(self, raceTrackLayout):
        # State Space
        # Same Number of States as there are positions on the racetrack
        # NumStates = raceTrack_Width * raceTrackHeight
        # Each State Contains:
        #   Positions (pos_x, pos_y)
        #   Velocity (vel_x, vel_y)
        # Define a Dictonary state_s
        #   Key: Position Tuple (pos_x, pos_y)
        #   Value: Combined Tuple (postion, velocity)
        
        #Velocity falls between [-5, 5]
        range_vel_x = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        range_vel_y = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        velCombos =  list(product(range_vel_x, range_vel_y))      
        stateSpace = {}
        for position in raceTrackLayout:
            for velocity in velCombos:
                pos_vel_tupleKey = (position, velocity)
                pos_x = position[0]
                pos_y = position[1]
                vel_x = velocity[0]
                vel_y = velocity[1]
                pos_vel_tupleValue = (pos_x, pos_y, vel_x, vel_y)
                stateSpace[pos_vel_tupleKey] = pos_vel_tupleValue
        return stateSpace
        
        
        
    