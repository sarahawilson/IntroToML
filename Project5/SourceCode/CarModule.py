# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

from typing import Tuple, Dict

class Car:
    def __init__(self, harshCrashLogic, raceTrackLayout):
        self.raceTrackLayout = raceTrackLayout
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
            self.curVelocity[0] = x_vel
        if abs(y_vel) <= 5:
            self.curVelocity[1] = y_vel
        
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
        #Only Run Bresenhams if there is a chagne in position, in order to check if along the line we hit a wall
        if(curPosition != nextPosition):
            positionsBetween = self._bresenhamPoints(curPosition, nextPosition)
                
            for linePos in positionsBetween:
                nextPossibeSpaceChar = self.raceTrackLayout[linePos]
                if ( nextPossibeSpaceChar == '#'):
                    wallHitOccured = True
                    break
            
            
        #We hit a wall
        if (wallHitOccured):
            if (self.harshCrashLogic):
                self.curPosition = self.startPosition
                self.curVelocity = self.startVelocity
            else:
                #TODO: Implement logic for starting at square closest to crash site
                self.curPosition = self.startPosition
                self.curVelocity = self.startVelocity
                
        #Did not hit a wall
        else:
            #Check if the next position is a finish line
            nextPosSpaceChar = self.raceTrackLayout[nextPosition]
            if(nextPosSpaceChar == 'F'):
                return True
            else:
                self.curPosition = nextPosition
        
        return False
        
        
        
        
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