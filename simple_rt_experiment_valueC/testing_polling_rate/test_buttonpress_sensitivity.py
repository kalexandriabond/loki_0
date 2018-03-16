from psychopy.iohub.client import launchHubServer
from psychopy import core, event
io=launchHubServer()

io_buttons = []
button_clock = core.Clock()

print("PRESS BUTTON NOW.")
core.wait(10)
button_clock.reset()
while button_clock.getTime() < 5:
    buttons = io.devices.keyboard.state #check constantly
    io_buttons.append(buttons)
    print(buttons)

io.quit()

print('button presses detected ', len(io_buttons))
