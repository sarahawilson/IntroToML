# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

from typing import Tuple, Dict
import random

class Car:
    def __init__(self, harshCrashLogic, raceTrackLayout, raceTrackWidth, raceTrackHegiht):
        self.raceTrackLayout = raceTrackLayout
        self.raceTrackWidth = raceTrackWidth
        self.raceTrackHeight = raceTrackHegiht
        self.curPosition = None
        self.curVelocity = None
        self.startPosition = None
        self.startVelocity = None
        self.harshCrashLogic = harshCrashLogic
        
    def init_car_kinematics(self, raceTrackStartPos):
        # Defines the starting position of the car
        self.curPosition = raceTrackStartPos
        self.curVelocity = (0,0)
        self.startPosition = self.curPosition
        self.startVelocity = self.curVelocity
        
    def init_car_rando(self, raceTrackStartPos):
        rand_x = random.randint(0,(self.raceTrackWidth-1))
        rand_y = random.randint(0,(self.raceTrackHeight-1))
        self.curPosition = (rand_x,rand_y)
        self.curVelocity = (0,0)
        self.startPosition = raceTrackStartPos
        self.startVelocity = self.curVelocity
        
                
    
    def resetOnHarshCrash(self):
        self.position = self.startPosition
        self.velocity = self.startVelocity
        
    def resetOnEasyCrash(self):
        #TODO: Figure out the Logic here 
        self.position = self.startPosition
        self.velocity = self.startVelocity
        
    def applyAcceleartion(self, acc : Tuple):
        # Applies the acceleariton 
        # and returns the new velocity
        x_vel = self.curVelocity[0] + acc[0]
        y_vel = self.curVelocity[1] + acc[1]

        # Apply the Velocity Cap 
        if abs(x_vel) <= 5:
            updated_vel_x = x_vel
        else:
            updated_vel_x = self.curVelocity[0]

        if abs(y_vel) <= 5:
            updated_vel_y = y_vel
        else:
            updated_vel_y = self.curVelocity[1]
            
        self.curVelocity = (updated_vel_x, updated_vel_y)
        
    def applyVelocity_QTry(self):
        # Applies the Velocity 
        # and returns if we have 
        # Velocity is taken from the car's current velocity
        
        #Position 1
        curPosition = self.curPosition
        curVelocity = self.curVelocity
        zz_pos1 = curPosition
        zz_vel2 = curVelocity
        
        if (curPosition == (4,2)):
            test = 2
        
        wallHitOccured = False
        
        
        # Check that you're not currently occuping a wall space
        # Used in Value Iteration
        quickCheckOnWallChar = self.raceTrackLayout[curPosition]
        if ( quickCheckOnWallChar == '#'):
            wallHitOccured = True
        
        
        nextPosition_x = curPosition[0] + curVelocity[0]
        nextPosition_y = curPosition[1] + curVelocity[1]
        
        #Position 2
        nextPosition = (nextPosition_x, nextPosition_y)
        
        stationaryMove = False
        if(not wallHitOccured):
            if (curPosition == nextPosition):
                stationaryMove = True
        
        if(not stationaryMove):
            positionsBetween = self._bresenhamPoints(curPosition, nextPosition)
            positionsBetween.append(nextPosition)
            
            if(len(positionsBetween) == 1):
                if((nextPosition[0] >= self.raceTrackWidth) or ((nextPosition[1] >= self.raceTrackHeight))):
                    test =1
                if((nextPosition[0] < 0) or ((nextPosition[1] < 0))):
                    test =1 
                else:
                    nextPossibeSpaceChar1 = self.raceTrackLayout[nextPosition]
                    if ( nextPossibeSpaceChar1 == '#'):
                        wallHitOccured = True
                    
            if(not wallHitOccured):
                for linePos in positionsBetween:
                    #Add logic for if the position between is not on the race track
                    if((linePos[0] >= self.raceTrackWidth) or ((linePos[1] >= self.raceTrackHeight))):
                        continue
                    if((linePos[0] < 0) or ((linePos[1] < 0))):
                        continue
                    
                    nextPossibeSpaceChar = self.raceTrackLayout[linePos]
                    if ( nextPossibeSpaceChar == 'F'):
                        zz_pos2 = linePos
                        return True
                        
                    if ( nextPossibeSpaceChar == '#'):
                        wallHitOccured = True
                        break
                
        
        if (wallHitOccured):
            if (self.harshCrashLogic):
                self.curPosition = self.startPosition
                self.curVelocity = self.startVelocity
            else:
                #TODO: Figure out simple logic 
                self.curPosition = self.startPosition
                self.curVelocity = self.startVelocity
        else:
            self.curPosition = nextPosition
            zz_pos2 = self.curPosition
            test =1 
            
        
        
    def applyVelocity(self):
        # Applies the Velocity 
        # and returns if we have 
        # Velocity is taken from the car's current velocity
        
        #Position 1
        curPosition = self.curPosition
        curVelocity = self.curVelocity
        
        nextPosition_x = curPosition[0] + curVelocity[0]
        nextPosition_y = curPosition[1] + curVelocity[1]
        
        #Position 2
        nextPosition = (nextPosition_x, nextPosition_y)
        
        wallHitOccured = False
        lastNonWallPosition = None
        quickCheckOnWallChar = self.raceTrackLayout[curPosition]
        if ( quickCheckOnWallChar == '#'):
            wallHitOccured = True
            if (self.harshCrashLogic):
                nextPosition = self.startPosition
            else:
                #TODO: Implement logic for starting at square closest to crash site
                lastNonWallPosition = self.startPosition
                nextPosition= self.startPosition

        #Only Run Bresenhams if there is a chagne in position, in order to check if along the line we hit a wall
        if((curPosition != nextPosition) and (wallHitOccured != True)):
            positionsBetween = self._bresenhamPoints(curPosition, nextPosition)
            
            for linePos in positionsBetween:
                #Add logic for if the position between is not on the race track
                if((linePos[0] >= self.raceTrackWidth) or ((linePos[1] >= self.raceTrackHeight))):
                    continue
                if((linePos[0] < 0) or ((linePos[1] < 0))):
                    continue
                
                nextPossibeSpaceChar = self.raceTrackLayout[linePos]
                lastNonWallPosition = linePos
                if ( nextPossibeSpaceChar == '#'):
                    wallHitOccured = True
                    break
                        
        #We hit a wall
        if (wallHitOccured):
            if (self.harshCrashLogic):
                self.curPosition = self.startPosition
                self.curVelocity = self.startVelocity
            else:
                if(lastNonWallPosition == None):
                    test = 1
                self.curPosition = lastNonWallPosition
                self.curVelocity = (0,0)
                
        #Did not hit a wall
        else:
            if(nextPosition[0] < 0):
                test =1
            if(nextPosition[1] < 0):
                test =1
            #Check if the next position is a finish line
            nextPosSpaceChar = self.raceTrackLayout[nextPosition]
            if(nextPosSpaceChar == 'F'):
                return True
            else:
                self.curPosition = nextPosition
        
        return False
        
    def _bresenhamPoints(self, position_1, position_2):
        # Note this is the Bresenahms Points taken directly from 
        # Wiki-Pedia, the following code was sourced from:
        # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
        x1 = position_1[0]
        y1 = position_1[1]
        x2 = position_2[0]
        y2 = position_2[1]
        
        pointsBetween = []
        if abs(y2 - y1) < abs(x2 - x1):
            if (x1 > x2):
               pointsBetween =  self._handleSlopeLow(x2, y2, x1, y1)
            else:
               pointsBetween =  self._handleSlopeLow(x1, y1, x2, y2)
        else:
            if (y1 > y2):
                pointsBetween = self._handleSlopeHigh(x2, y2, x1, y1)
            else:
                pointsBetween = self._handleSlopeHigh(x1, y1, x2, y2)
                
        return pointsBetween
                
        
    def _handleSlopeLow(self, x1, y1, x2, y2):
        # Note this is the Bresenahms Points taken directly from 
        # Wiki-Pedia, the following code was sourced from:
        # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
        dx = x2 - x1 
        dy = y2 - y1
        y_increment = 1
        if (dy < 0):
            y_increment = -1
            dy = -dy
        D = (2*dy) - dx
        y = y1
        
        pointsBetween = []
        for x in range(x1,x2,1):
            point = (x,y)
            pointsBetween.append(point)
            if (D > 0 ):
                y = y + y_increment
                D = D + (2 * (dy - dx))
            else:
                D = D + (2*dy)
        return pointsBetween
      
    def _handleSlopeHigh(self, x1, y1, x2, y2):
        # Note this is the Bresenahms Points taken directly from 
        # Wiki-Pedia, the following code was sourced from:
        # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
        dx = x2 - x1
        dy = y2 - y1
        x_increment = 1
        if (dx < 0):
            x_increment = -1
            dx = -dx
        D = (2 * dx) - dy
        x = x1

        pointsBetween = []
        for y in range(y1,y2,1):
            point = (x,y)
            pointsBetween.append(point)
            if (D > 0 ):
                x = x + x_increment
                D = D + (2 * (dx - dy))
            else:
                D = D + (2*dx)
        return pointsBetween