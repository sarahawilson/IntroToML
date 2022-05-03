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
                    qTable_of_A[acceleration] = -100000
                #append to overall q table
                qTable_of_SA[pos_vel_tupleKey] = qTable_of_A
        return qTable_of_SA
    
    
    def runTrain(self, epsilon, learningRate, discount, iterations):
        
        # Set up the Car
        # In Q-Learning the Car Explores the Space to determine the policy
        curCar = CarModule.Car(self.harshCrashLogic, self.raceTrack.raceTrackLayout, self.raceTrack.width, self.raceTrack.height)
        curCar.init_car_kinematics(self.raceTrack.startPosition)
        
        # Training Process runs until the car reaches the finish line
        # or number of iterations has been reached
        reachedFinishLine = False
        loopIterations = 0
        while ((not reachedFinishLine) and (loopIterations < iterations)):
            
            q_s_AccessKey = (curCar.curPosition, curCar.curVelocity)
            cur_q_a_values_basedOnCarState = self.q_table[q_s_AccessKey]
            
            #Let the greedy epsilon algorithm pick the acceleration to use
            ep_picked_acceleration = self._greedy_epsilon(cur_q_a_values_basedOnCarState, epsilon)
            
            
            
            
            
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
    

