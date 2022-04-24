# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

import matplotlib.pyplot as plt
from matplotlib import colors as clrs
import numpy as np

class RaceTrackHelper:
    def __init__(self, filePath):
        self.filePath = filePath
        self.height = 0
        self.width = 0
        self.raceTrack = None
        
    def makeRaceTrack(self):
        curTrack = Track(self.filePath)
        self.raceTrack = curTrack
        return curTrack
    
    def displayRaceTrack(self):
        
        
        #Make a Map for Plotting based on the Track 
        colorTrackData = []
        rowSpaceColorList = []
        for row in self.raceTrack.spaces:
            for singleSpace in row:
                curSpaceColorValue = singleSpace.cmapValue
                rowSpaceColorList.append(curSpaceColorValue)
            colorTrackData.append(rowSpaceColorList)
            rowSpaceColorList = []
        
        colorTrackDataArray = np.array(colorTrackData)
        
        #FLip the Array about the Y Axis to perseve the map when plotted
        colorTrackDataArrayFlipped = np.flip(colorTrackDataArray, 0)
        
        custom_cMap = clrs.ListedColormap(['b','w','g', 'r'])
        
        fig, ax = plt.subplots()
        c = ax.pcolor(colorTrackDataArrayFlipped, edgecolors='k', linestyle= 'solid', linewidths=1, cmap=custom_cMap, vmin=0.0, vmax=4.0)
        
        # Show the Index in the middle
        ax.set_yticks(np.arange(self.height) + 0.5, minor=False)
        ax.set_xticks(np.arange(self.width) + 0.5, minor=False)
        
        # Set the Labels as the row col idx
        xticklabels = range(self.height) # could be text
        yticklabels = range(self.width) # could be text 
        
        ax.set_xticklabels(xticklabels, minor=False)
        ax.set_yticklabels(yticklabels, minor=False)
        
        plt.show()

        
        
    
class Track:
    def __init__(self, filePath):
        self.filePath = filePath
        self.trackSize = self._getTrackSize()
        self.trackHeight = self.trackSize[0]
        self.trackWeight = self.trackSize[1]
        self.trackLayoutRaw = self._getTrackLayout()
        self.spaces = self._makeSpaces()
        
    def _getTrackSize(self):
        #Gets the Track Size (row / height, col/width)
        #from the .txt file
        with open(self.filePath, 'r') as trackFile:
            lines = trackFile.readlines()
        trackFile.close()
        sizeLine = lines[0].split(',')
        height = int(sizeLine[0])
        width = int(sizeLine[1])
        return (height,width)
     
    def _getTrackLayout(self):
        #Gets the track layout (wall, start, finish, track)
        #from the .txt file
        with open(self.filePath, 'r') as trackFile:
            lines = trackFile.readlines()
        trackFile.close()
        layout = lines[1:]
        return layout

    def _makeSpaces(self):
        spaceList = []
        rowSpaceList = []
        xIdx = 0
        yIdx = 0
        for rowLine in self.trackLayoutRaw:
            rowLine = rowLine.rstrip('\n')
            for charValue in rowLine:
                if (charValue == '#'):
                    #Wall Space
                    spaceColor = 'black'
                    cmapValue = 0 
                    
                elif (charValue == '.'):
                    #Track Space
                    spaceColor = 'white'
                    cmapValue = 1 
                    
                elif (charValue == 'S'):
                    #Starting Line
                    spaceColor = 'green'
                    cmapValue = 2 
                    
                elif (charValue == 'F'):
                    #Finish Line
                    spaceColor = 'red'
                    cmapValue = 3 
                
                curSpace = self.Space(xIdx, yIdx, spaceColor, cmapValue)
                rowSpaceList.append(curSpace)
                yIdx = yIdx + 1
            spaceList.append(rowSpaceList)
            rowSpaceList = []
            xIdx = xIdx + 1 
        return spaceList
        
    class Space:
        def __init__(self, x, y, color, cmapValue):
            self.x = x
            self.y = y
            self.color = color
            self.cmapValue = cmapValue
