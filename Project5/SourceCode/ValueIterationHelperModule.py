# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

from typing import Tuple, Dict
import ImportHelperModule

class ValIterHelper:
    def __init__(self, RaceTrack, harshCrashLogic):
        self.harshCrashLogic = harshCrashLogic
        self.raceTrack = RaceTrack
        self.rewards = RewardMap()
        self.values = ValueMap()
        self.actions = ActionMap()
        self.car = Car()

    def setInitialConditions(self):
        # Set the Inital Conditions to run the Value Iteration Algo
        # Sets the Rewards based on the Race Track
        # Sets the Initial Value Map
        self.rewards.defineRewards(self.raceTrack)
        self.values.defineInitialValueMap(self.raceTrack)
        
        # Get the Start Position from the RaceTrack
        startPos = self.raceTrack.findStartPosition()
        self.car.defineStartingMotion(startPos)
        
    def clearAllAndResetInitialConditions(self):
        # Clears and cleans the value iteration class 
        # in prep for the next run
        self.rewards = None
        self.rewards = RewardMap()
        self.values = None
        self.values = ValueMap()
        self.setInitialConditions()
        self.car = None
        self.car = Car()
        self.car.defineStartingMotion()
    
    def run(self, discount, epsilon):
        # Runs the Value Iteration Algorithm 
        timeStep = 0
        
        deltaMaxValues = epsilon + 1
        
        while (deltaMaxValues > epsilon):
            timeStep = timeStep + 1
            
            #For all s in S
            for position_s in self.raceTrack.race_track:
                #For all a in A
                for action_a in self.actions.actionMap:
                    curReward = self.rewards.rewardMap[position_s]
                    statePrime = _genNextState(position_s, )
            
            
            #Ensure the While Loop Exits
            deltaMaxValues = epsilon - 0.1
            
    def _getNextState(self, position_s : Tuple, velocity : Tuple, acc_toApply_a : Tuple):
        
        nextState_Position = None
        nextState_Velocity = None
        
        
        # Get the space from the racetrack
        curSpaceChar = self.raceTrack.race_track[position_s]
        
        # First check if the position_s is a wall
        # If a wall and harsh Crash
        # Next state will be the starting position
        if ( curSpaceChar == '#'):
            if (self.harshCrashLogic):
                nextState_Position = self.car.startPosition
                nextState_Velocity = self.car.startVelocity
            else:
                #TODO: Implement logic for starting at square closest to crash site
                nextState_Position = self.car.startPosition
                nextState_Velocity = self.car.startVelocity
        
        # Next check if the space is a starting square or a track square
        # Use Bresenhams Algorithm to determine if you hit a wall at
        # any point along your line of travel
        elif ( curSpaceChar == 'S' or curSpaceChar == '.' ):
            # Determine the Stopping Position (pos_x_2, pos_y_2)
            # Given an Acceleration (Aciton a)
            
            acc_x_t  = acc_toApply_a[0]
            vel_x_t = velocity[0] + acc_x_t
            #Cap the Velocities
            if (vel_x_t < -5):
                vel_x_t = -5
            elif (vel_x_t > 5):
                vel_x_t = 5
            pos_x_t = position_s[0] + vel_x_t
            
            acc_y_t = acc_toApply_a[1]
            vel_y_t = velocity[1] + acc_y_t
            #Cap the Velocities
            if (vel_y_t < -5):
                vel_y_t = -5
            elif (vel_y_t > 5):
                vel_y_t = 5
            pos_y_t = position_s[1] + vel_y_t
            
            #(pos_x_2, pos_y_2)
            kinematicNextPosition = (pos_x_t, pos_y_t)
            kinematicNextVelocity = (vel_x_t, vel_y_t)
            
            wallHitOccured = False
            #Only Run Bresenhams if there is a chagne in position, in order to check if along the line we hit a wall
            if((kinematicNextPosition[0] != position_s[0]) and (kinematicNextPosition[1] != position_s[1])):
                positionsBetween = self._bresenhamPoints(position_s, kinematicNextPosition)
                
                for linePos in positionsBetween:
                    nextPossibeSpaceChar = self.raceTrack.race_track[linePos]
                    if ( nextPossibeSpaceChar == '#'):
                        wallHitOccured = True
                        break
            
            #Handle the Wall Hit
            if (wallHitOccured):
                if (self.harshCrashLogic):
                    nextState_Position = self.car.startPosition
                    nextState_Velocity = self.car.startVelocity
                else:
                    #TODO: Implement logic for starting at square closest to crash site
                    nextState_Position = self.car.startPosition
                    nextState_Velocity = self.car.startVelocity
                
            else:
                nextState_Position = kinematicNextPosition
                nextState_Velocity = kinematicNextVelocity
                
        #Finally Check for Finish
        #TODO: need to think through if this is the right way to do this
        elif ( curSpaceChar == 'F'):
            nextState_Position = None
            nextState_Velocity = None 
                
        return (nextState_Position, nextState_Velocity)   
            
      
    def _bresenhamPoints(self, position_1, position_2):
        pointsBetween = []
        x_between = []
        y_between = []
        
        x1 = position_1[0]
        y1 = position_1[1]
        x2 = position_2[0]
        y2 = position_2[1]
        
        x, y = x1,y1
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        grad = dy/float(dx)
        
        if grad > 1:
            dx, dy = dy, dx
            x, y = y, x
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        
        p = 2 * dy - dx
        
        x_between.append(x)
        y_between.append(y)
        pointsBetween.append(x,y)
        
        for val in range(dx):
            if p > 0:
                y = y + 1 if y < y2 else y -1
                p = p + 2*(dy - dx)
            else:
                p = p + 2*dy
            
            x = x + 1 if x < x2 else x -1
            x_between.append(x)
            y_between.append(y)
            pointsBetween.append(x,y)
        
        return pointsBetween
            
        

class Car:
    def __init__(self):
        self.position = None
        self.velocity = None
        self.startPosition = None
        self.startVelocity = None
        
    def defineStartingMotion(self, raceTrackStartPos : Tuple):
        # Defines the starting position of the car
        self.position = raceTrackStartPos
        vel_x = 0
        vel_y = 0
        self.velocity = (vel_x, vel_y)
        self.startPosition = raceTrackStartPos
        self.startVelocity = (vel_x, vel_y)
        
    def getCurrentPosition(self):
        return self.position
    
    def getCurrentVelocity(self):
        return self.velocity
    
    def resetMotionHarshCrash(self):
        self.position = self.startPosition
        self.velocity = self.startVelocity
        

class RewardMap:
    def __init__(self):
        self.rewardMap = None
    
    def defineRewards(self, RaceTrack):
        # Defines a Rewards Dictonary based on the 
        # layout of the race track
        # All spaced except for the finish have a reward (or cost) of -1
        rewardsDict = {}
        for position in RaceTrack.race_track:
            spaceChar = RaceTrack.race_track[position]
            if (spaceChar != 'F'):
                rewardsDict[position] = -1
            else:
                rewardsDict[position] = 0
            
        self.rewardMap = rewardsDict        
        
class ValueMap:
    def __init__(self):
        self.valueMap = None
    
    def defineInitialValueMap(self, RaceTrack):
        # Defines the Initial Value Map based on 
        # layout of the race track
        # Initially will all be set to zeros
        valueDict = {}
        for position in RaceTrack.race_track:
            valueDict[position] = 0
            
        self.valueMap = valueDict
        
class ActionMap:
    def __init__(self):
        self.actionMap = self.defineActions()
        self.propApplied = 0.8
        self.propNotApplied = 0.2
        
    def defineActions(self):
        # Defines the set of all possible actions
        # Tuple (acc_x, acc_y)
        # There should be 9 unique tuples of acc_x and acc_y
        # acc_x and acc_y can be -1, 0, 1
        actionList = []
        actionList.append((-1,-1))
        actionList.append((-1,-0))
        actionList.append((-1,-1))
        
        actionList.append((0,-1))
        actionList.append((0,-0))
        actionList.append((0,-1))
        
        actionList.append((1,-1))
        actionList.append((1,-0))
        actionList.append((1,-1))
        
        return actionList
        
        
        
        
        
        
    