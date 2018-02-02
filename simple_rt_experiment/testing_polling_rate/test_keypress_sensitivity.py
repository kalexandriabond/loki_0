from psychopy.iohub.client import launchHubServer
from psychopy import core, event
import numpy as np
import pyautogui

io=launchHubServer()

io_keyboard = []


print("PRESS KEY NOW.")

key_clock = core.Clock()
key_clock.reset()
while key_clock.getTime() < 5:
    # pyautogui.down('d')
    keys = io.devices.keyboard.state #check constantly
    io_keyboard.append(keys)
pyautogui.keyUp('d')

io_buttons = []
button_clock = core.Clock()

np.savetxt("key_polling_rate.csv", io_keyboard, delimiter=',',comments='')

print("PRESS BUTTON NOW.")
core.wait(1)
button_clock.reset()
while button_clock.getTime() < 5:
    buttons = io.devices.keyboard.state #check constantly
    io_buttons.append(buttons)
    print(buttons)

io.quit()

print('button presses detected ', len(io_buttons))
print('key presses detected ', len(io_keyboard))

np.savetxt("button_polling_rate.csv", str(button_keyboard), delimiter=',',comments='')


#so far, there are slightly more key presses detected
#than button presses...

# print('keys ', io_keyboard)
# print('buttons ', buttons)
