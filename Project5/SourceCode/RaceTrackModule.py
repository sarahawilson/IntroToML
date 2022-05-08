# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

from typing import Tuple, Dict

class RaceTrack:
    #Object that represetnts the race track
    def __init__(self, raceTrackLayout : Dict, height : int, width : int):
        self.raceTrackLayout = raceTrackLayout
        self.startPosition = self._findStartPosition()
        self.endPositions = self._findEndPositions()
        self.height = height
        self.width = width
        
    def _findStartPosition(self):
        #Finds the starting position based on which squares have the S symbol
        startPos = None
        for pos in self.raceTrackLayout:
            spaceChar = self.raceTrackLayout[pos]
            if(spaceChar == 'S'):
                startPos = pos
                break
        return startPos
    
    def _findEndPositions(self):
        #Finds the ending positions based on which squares have the F symbol
        endPositions = []
        for pos in self.raceTrackLayout:
            spaceChar = self.raceTrackLayout[pos]
            if(spaceChar == 'F'):
                endPositions.append(pos)
        
        return endPositions

