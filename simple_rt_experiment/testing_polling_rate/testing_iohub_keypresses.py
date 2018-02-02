from psychopy.iohub.client import launchHubServer
from psychopy import core
# Start the ioHub process. 'io' can now be used during the
# experiment to access iohub devices and read iohub device events.
io=launchHubServer()
clock = core.Clock()
print "Press any Key to Exit Example....."
while clock.getTime() < 30:
# Wait until a keyboard event occurs
    keys = io.devices.keyboard.getKeys(keys=['d',],clear=False) #check constantly


    if not keys:
        print(len(keys))
    else:
        keyname = keys[-1]
        keyout = keyname.key
        print(len(keyout))

print("Exiting experiment....")
# Stop the ioHub Server
io.quit()
