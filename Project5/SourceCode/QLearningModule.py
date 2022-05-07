# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

from typing import Tuple, Dict
from itertools import product
import copy
import numpy as np
import random

import RaceTrackModule
import CarModule

class QLearnHelper:
    def __init__(self, RaceTrack, harshCrashLogic):
        self.raceTrack = RaceTrack
        self.actionSpace = ActionSpace_A()
        self.q_table = self._init_q_table()
        self.harshCrashLogic = harshCrashLogic

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
        for position in self.raceTrack.raceTrackLayout:
            for velocity in velCombos:
                pos_vel_tupleKey = (position, velocity)
                
                #Now build the q table related to accerelation
                for acceleration in accCombos:
                    qTable_of_A[acceleration] = -1000
                    #qTable_of_A[acceleration] = random.random()
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
    
    
    def runTrain(self, epsilon, learningRate, discount, iterations):
       
        # Set up the Car
        # In Q-Learning the Car Explores the Space to determine the policy
        curCar = CarModule.Car(self.harshCrashLogic, self.raceTrack.raceTrackLayout, self.raceTrack.width, self.raceTrack.height)
        curCar.init_car_kinematics(self.raceTrack.startPosition)
        
        zz_curPos = curCar.curPosition
        zz_curVel = curCar.curVelocity
        
        # Training Process runs until the car reaches the finish line
        # or number of iterations has been reached
        reachedFinishLine = False
        loopIterations = 0
        while ((not reachedFinishLine) and (loopIterations < iterations)):
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
            reachedFinishLine = curCar.applyVelocity_QTry()
            
            zzn_curPos = curCar.curPosition
            zzn_curVel = curCar.curVelocity
            
            if (not reachedFinishLine):
                curPos = curCar.curPosition
                curVel = curCar.curVelocity
                
                # Get Q(s')
                q_s_primeAccessKey = (curPos, curVel)
                q_s_prime = self.q_table[q_s_primeAccessKey]
                
                # Determine Max a' based on Q(s')
                #max_q_acc_prime = np.max(q_s_prime)
                max_q_acc_prime_AccessKey = max(q_s_prime, key=q_s_prime.get)
                #print(max_q_acc_prime_AccessKey)
                max_q_acc_prime = q_s_prime[max_q_acc_prime_AccessKey]
                
                # Update the q_s_a values
                reward = -1
                cur_q_a_values_basedOnCarState[accToApply] = cur_q_a_values_basedOnCarState[accToApply] + (learningRate *(reward + (discount * max_q_acc_prime) - q_s_a_value))
            
            loopIterations = loopIterations + 1
            epsilon = epsilon * 0.999999
            
        # Vary the Learning Rate (Variable Learning)
#        epsilon = epsilon * 0.999999
#        if (learningRate > 0.01):
#            learningRate = learningRate * 0.8
            
                
    def runTest(self, epsilon, iterations):
        # Runs the Test on the Car in the Race Track based on the results learned in the training
         
        #Set up the Car 
        curCar = CarModule.Car(self.harshCrashLogic, self.raceTrack.raceTrackLayout, self.raceTrack.width, self.raceTrack.height)
        curCar.init_car_kinematics(self.raceTrack.startPosition)
        print('--Testing the Race Car--')
        if(self.harshCrashLogic):
            print('Bad Crash')
        else:
            print('Simple Crash')
        print('Car Starting Position:')
        print('Car Position' + str(curCar.curPosition))
        print('Car Velocity' + str(curCar.curVelocity))
        
        # Let the car run around the track until it has reached the finish line
        reachedFinishLine = False
        
        elapsedTime = 0 
        loopIterations = 0
        while ((not reachedFinishLine) and (loopIterations < iterations)):
            q_s_AccessKey = (curCar.curPosition, curCar.curVelocity)
            cur_q_a_values_basedOnCarState = self.q_table[q_s_AccessKey]
            
            #Let the greedy epsilon algorithm pick the acceleration to use
            ep_picked_acceleration = self._greedy_epsilon(cur_q_a_values_basedOnCarState, epsilon)
            
            print('Car Acceleration:' + str(ep_picked_acceleration))
            
            # Apply Acceleration to Car
            #Apply Acceleariton First (changes velocity)
            curCar.applyAcceleartion(ep_picked_acceleration)
            print('Car Next State:')
            print('Car Position' + str(curCar.curPosition))
            print('Car Velocity' + str(curCar.curVelocity))
            
            elapsedTime = elapsedTime + 1
            loopIterations = loopIterations + 1
            #Then Apply the Velocity (changes position) 
            reachedFinishLine = curCar.applyVelocity_QTry()
            
        return elapsedTime    
        
class ActionSpace_A:
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
    

