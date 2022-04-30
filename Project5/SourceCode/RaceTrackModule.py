# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

from typing import Tuple, Dict

class RaceTrack:
    def __init__(self, raceTrackLayout : Dict, height : int, width : int):
        self.raceTrackLayout = raceTrackLayout
        self.startPosition = self._findStartPosition
        self.height = height
        self.width = width
        
    def _findStartPosition(self):
        startPos = None
        for pos in self.raceTrackLayout:
            spaceChar = self.raceTrackLayout[pos]
            if(spaceChar == 'S'):
                startPos = pos
                break
        return startPos

