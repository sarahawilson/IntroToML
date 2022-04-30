# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

from typing import Tuple, Dict


class ImportHelper:
    def __init__(self, filePath):
        self.filePath = filePath
        
        
    def LoadRaceTrack(self):
        # Loads the Rack Track File 
        # Returns a Dictonary that is the Race Track
        with open(self.filePath, 'r') as trackFile:
            lines = trackFile.readlines()
        trackFile.close()
        sizeLine = lines[0].split(',')
        height = int(sizeLine[0])
        width = int(sizeLine[1])
        trackLayout = lines[1:]
        #trackLayoutFlipped = trackLayout.reverse()
        
        pos_x = 0
        pos_y = 0
        raceTrackDict = {}
        for rowLine in trackLayout[::-1]:
            rowLine = rowLine.rstrip('\n')
            for charValue in rowLine:
                raceTrackDict[(pos_x, pos_y)] = charValue
                pos_y = pos_y + 1
            pos_x = pos_x + 1 
        
        genRaceTrack = RaceTrack(raceTrackDict, height, width)
        
        return genRaceTrack
    
    
class RaceTrack:
    def __init__(self, race_track : Dict, height : int, width : int):
        self.race_track = race_track
        self.height = height
        self.width = width
        
    def findStartPosition(self):
        startPos = None
        for pos in self.race_track:
            spaceChar = self.race_track[pos]
            if(spaceChar == 'S'):
                startPos = pos
                break
        return startPos