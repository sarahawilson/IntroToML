# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

from typing import Tuple, Dict
from itertools import product
import copy
import random

import RaceTrackModule
import CarModule


class ValIterHelper:
    def __init__(self, RaceTrack, harshCrashLogic):
        self.raceTrack = RaceTrack
        self.actionSpace = ActionSpace_A()
        self.stateSpace = StateSpace_S(self.raceTrack.raceTrackLayout)
        self.q_table = self._init_q_table()
        self.value_table = self._init_value_table()
        self.policy_table = self._init_policy_table()
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
        
    def runTrain(self, epsilon, iterations, discount):
        
        curCar = CarModule.Car(self.harshCrashLogic, self.raceTrack.raceTrackLayout, self.raceTrack.width, self.raceTrack.height)
        curCar.init_car_kinematics(self.raceTrack.startPosition)
        
        loopIterations = 0
        reachedConvergance = False
        while ((not reachedConvergance) and (loopIterations < iterations)):
            
            prev_ValueTable = copy.deepcopy(self.value_table)
            overall_delta_v = 0
            
            #For all s in S
            for pos_vel_state in self.stateSpace.states_S:
                #For all a in A
                
                max_q_val = -10000
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
        
        print('--Training the Race Car--')
        if(self.harshCrashLogic):
            print('Bad Crash')
        else:
            print('Simple Crash')
        print('Reached Convergance')
        print('Asked for Iterations:' + str(iterations))
        print('Ran Iterations:'  + str(loopIterations))
        print('Reached Convergance by Epsilon:' +str(reachedConvergance))
        print('-------------------------')
        
        
    def runTest(self):
        
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
            
            #Have the car perfrom the aciton given the policy
            curCar.applyAcceleartion(bestPolicyAction)
            print('Car Next State:')
            print('Car Position' + str(curCar.curPosition))
            print('Car Velocity' + str(curCar.curVelocity))
            
            elapsedTime = elapsedTime + 1
            reachedFinish = curCar.applyVelocity()
        
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
        
        
class StateSpace_S:
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
        
        
        
    