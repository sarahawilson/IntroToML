# Sarah Wilson
# Project 5 - Renforcment Learning 
# Race Track - Zoom Zoom

import RaceTrackHelperModule

# User Defined Varaibles 
trackName = 'O'
algorithm = 'Q'

 

# Init based on the user defined varaibles 
if(trackName == 'L'):
    raceTrackFile = r'C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\Project5\SourceCode\Tracks\L-track.txt'
elif(trackName == 'O'):
    raceTrackFile = r'C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\Project5\SourceCode\Tracks\O-track.txt'
elif(trackName == 'R'):
    raceTrackFile = r'C:\Users\Sarah Wilson\Desktop\JHU Classes\IntroToML\Project5\SourceCode\Tracks\R-track.txt'

# Load and Build the Track 
rtHelper = RaceTrackHelperModule.RaceTrackHelper(raceTrackFile)
rtHelper.makeRaceTrack()
rtHelper.displayRaceTrack()

# Run the Algoirthm 
