# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom


import ImportHelperModule

class ValIterHelper:
    def __init__(self, RaceTrack):
        self.raceTrack = RaceTrack
        self.rewards = RewardMap()
        self.values = ValueMap()
        self.actions = ActionMap()

    def setInitialConditions(self):
        # Set the Inital Conditions to run the Value Iteration Algo
        # Sets the Rewards based on the Race Track
        # Sets the Initial Value Map
        self.rewards.defineRewards(self.raceTrack)
        self.values.defineInitialValueMap(self.raceTrack)
        
    def clearAllAndResetInitialConditions(self):
        # Clears and cleans the value iteration class 
        # in prep for the next run
        self.rewards = None
        self.rewards = RewardMap()
        self.values = None
        self.values = ValueMap()
        self.setInitialConditions()
    
    def run(self, discount, epsilon):
        # Runs the Value Iteration Algorithm 
        timeStep = 0
        
        deltaMaxValues = epsilon + 1
        
        while (deltaMaxValues > epsilon):
            timeStep = timeStep + 1
            
            #For all s in S
            for position in self.raceTrack.race_track:
                #For all a in A
                for action in self.actions.actionMap:
                    test = 1
            
            
            #Ensure the While Loop Exits
            deltaMaxValues = epsilon - 0.1
        


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
        
        
        
        
        
        
    