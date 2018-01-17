
import os,random,sys
from psychopy import visual,core,event,monitors
import pandas as pd
import numpy as np


"""set data path & collect information from experimenter"""
testing = int(raw_input("Testing? "))
if testing is not 1 and testing is not 0:
    sys.exit('Enter 0 or 1.')

image_directory = os.getcwd() + '/images/'
exp_param_directory = os.getcwd() + '/experimental_parameters/'
analysis_directory = os.getcwd() + '/analysis/'

if testing:
    subj_id = 'test'
    condition = str(0)
    session = str(0)
    exp_param_file = exp_param_directory + 'v5.csv'
else:
    subj_id = raw_input("Subject ID: ")
    condition = raw_input("Condition: ")
    session = raw_input("Session: ")

    if condition == '1':
        exp_param_file = exp_param_directory + 'v5.csv'
    elif condition == '2':
        exp_param_file == exp_param_directory + 'highV_rp.csv'
    else:
        sys.exit("Unknown condition.")

file_name = subj_id + "_cond" + condition + "_session" + session
data_path = os.getcwd() + '/data/' + file_name + ".csv"

if not testing and os.path.exists(data_path):
    sys.exit(file_name + " already exists!")


instructions = ("In this task, you will have a choice between two slot machines represented by fractal images. When you choose one of these images, you will lose or win points. Choosing the same slot machine will not always give you the same points, but one is better than the other. After making your choice, you will receive feedback about the outcome. Your goal is to choose the machine that gives the greatest reward. "
+"\n\nChoose the left target by pressing the left arrow key and choose the right target by pressing the right arrow key. Press any key when you're ready to begin.")
slow_trial = ("Too slow! \nChoose quickly.")
fast_trial = ("Too fast! \nSlow down.")

"""initialize dependent variables & images"""
rt_list = []
choice_list = []

images = []
for file in os.listdir(image_directory):
    if file.lower() == ("fractal_2.png")  or file.lower() == ("fractal_3.png") :
        images.append(image_directory+file)

"""instantiate psychopy object instances"""
clock = core.Clock()

mbp_monitor = monitors.Monitor('mbp_15_inch')
mbp_monitor.setSizePix = [1440,900]
mbp_monitor.saveMon()

screen_size = mbp_monitor.setSizePix
# screen_size=[800,600]
center=[0,0]

if screen_size != mbp_monitor.setSizePix:
    center[0] = (mbp_monitor.setSizePix[0]/2) - (screen_size[0]/2)
    center[1] = (mbp_monitor.setSizePix[1]/2) - (screen_size[1]/2)

window = visual.Window(size = screen_size, units='pix', monitor = mbp_monitor, color = [-1,-1,-1], \
       colorSpace = 'rgb', blendMode = 'avg', useFBO = True, allowGUI = \
       False,fullscr=True, pos=center)

inst_msg = visual.TextStim(win=window, units='pix',antialias='False', text=instructions, wrapWidth=screen_size[0], height=screen_size[1]/15)
speed_msg = visual.TextStim(win=window, units='pix',antialias='False', text=slow_trial, pos = [0,screen_size[1]/2], height=screen_size[0]/12,
alignHoriz='center', wrapWidth=screen_size[1]*2, colorSpace='rgb',color=[1,0,0], bold=True)
choice_emphasis = visual.Rect(win=window, units='pix', height = screen_size[0]/3, width= screen_size[0]/3, lineColorSpace='rgb',lineColor=[1,1,1], lineWidth=5)
HighValueFractal = visual.ImageStim(window, image=images[1],units='pix',size=[screen_size[0]/5])
LowValueFractal = visual.ImageStim(window, image=images[0],units='pix',size=[screen_size[0]/5])
rewardMsg = visual.TextStim(win=window,units='pix',antialias='False',pos=[-10,0], colorSpace='rgb', color=[1,1,1],height=screen_size[0]/10)

"""specify constants"""
exp_param = pd.read_csv(exp_param_file, header=0)
exp_param.columns = ['t1_r', 't2_r', 'ssd', 'cp', 'r_diff', 'mu_r_diff', 'sigma']
reward_t1 = np.round(exp_param.t1_r,2)
reward_t2 = np.round(exp_param.t2_r,2)
rewards = np.transpose(np.array([reward_t1, reward_t2]))
max_reward_idx = np.argmax(rewards,1)



max_rt = .7
min_rt = .08
left_pos = [-screen_size[0]/2,0]
right_pos = [screen_size[0]/2,0]
fb_time = 0.7
n_trials = len(exp_param.cp)
n_test_trials = 10

iti_min = .1
iti_max = .7
iti_list = []

if testing:
    n_trials = n_test_trials

cp_list = exp_param.cp.values[0:n_trials].tolist()
solutions = max_reward_idx[0:n_trials]

"""define forced choice trial type"""
def ForcedChoiceTrial(max_reward_idx):
    if max_reward_idx == 0:
        HighValueFractal.setPos(left_pos)
        LowValueFractal.setPos(right_pos)

    elif max_reward_idx == 1:
        HighValueFractal.setPos(right_pos)
        LowValueFractal.setPos(left_pos)

    HighValueFractal.setAutoDraw(True)
    LowValueFractal.setAutoDraw(True)

    iti = random.uniform(iti_min, iti_max)
    iti_list.append(iti)
    core.wait(iti)

    window.flip()
    clock.reset()
    data = event.waitKeys(keyList=['left','right','escape'],timeStamped=clock)

    choice=data[0][0]
    rt=data[0][1]

    if rt >= max_rt:
        speed_msg.text = slow_trial
        speed_msg.draw()
    elif rt <= min_rt:
        speed_msg.text = fast_trial
        speed_msg.draw()

    return choice,rt


"""give instructions"""
instruction_phase = True
while instruction_phase:
    inst_msg.setAutoDraw(True)
    inst_keys = event.getKeys()
    if 'escape' in inst_keys:
        sys.exit('escape key pressed.')
    if len(inst_keys) > 0:
        instruction_phase = False
    if event.getKeys(keyList=['escape']):
        sys.exit('escape key pressed.')
    if instruction_phase:
        window.flip()

inst_msg.setAutoDraw(False)
window.flip()

""""present choices"""
for i in range(0,n_trials):
    HighValueFractal.setAutoDraw(False)
    LowValueFractal.setAutoDraw(False)
    window.flip()

    choice,rt = ForcedChoiceTrial(max_reward_idx[i])
    rt_list.append(rt)

    if choice == 'left':
        choice_list.append(0)
        rewardMsg.text = str(reward_t1[i])
        choice_emphasis.pos = left_pos

    elif choice == 'right':
        choice_list.append(1)
        rewardMsg.text = str(reward_t2[i])
        choice_emphasis.pos = right_pos

    if choice == 'escape':
        sys.exit('escape key pressed.')

    choice_emphasis.draw()
    rewardMsg.draw()
    window.flip()
    core.wait(fb_time)

accuracy_list = np.equal(solutions, np.asarray(choice_list))
accuracy_list.tolist()

"""save data"""
header = ("choice, rt, cp, accuracy")
dvs = np.transpose(np.array([choice_list, rt_list, cp_list, accuracy_list]))
np.savetxt(data_path, dvs, header = header, delimiter=',',comments='')

window.close()
