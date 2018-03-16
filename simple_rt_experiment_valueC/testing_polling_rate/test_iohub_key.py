
import os,sys,random
from psychopy import prefs
prefs.general['audioLib'] = ['pygame']
from psychopy import visual,core,event,monitors,sound,info
from pandas import read_csv
from psychopy.iohub.client import launchHubServer

import numpy as np
from random import shuffle
io=launchHubServer()
window = visual.Window(size = [1920,1080], units='pix', color = [-1,-1,-1], \
       colorSpace = 'rgb', blendMode = 'avg', useFBO = True, allowGUI = \
       False,fullscr=True)

# Assumes 'io' object was created using the
# psychopy.iohub.launchHubProcess() function and
# 'window' is a full screen PsychoPy Window

# save some 'dots' during the trial loop
keyboard = io.devices.keyboard

# Store the RT calculation here
spacebar_rt=0.0

# build visual stim as needed
# ....

# Display first frame of screen
flip_time=window.flip()

io.clearEvents('all')

# Run each trial until space bar is pressed
while spacebar_rt == 0.0:
    events=keyboard.getEvents()

    for kb_event in events:
        if kb_event.key == ' ':
            spacebar_rt=kb_event.time-flip_time

    # Update visual stim as needed
    # ....

    # Display next frame of screen
    window.flip()
