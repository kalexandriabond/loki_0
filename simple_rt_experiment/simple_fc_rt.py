
import os,random,sys
from psychopy import visual,core,event,monitors
import pandas as pd
import numpy as np
from random import shuffle


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
    exp_param_file = exp_param_directory + 'high_volatility_high_conflict.csv'
else:
    subj_id = raw_input("Subject ID: ")
    condition = raw_input("Condition: ")
    session = raw_input("Session: ")

    if condition == 'lvlc':
        exp_param_file = exp_param_directory + 'lv_lc0.csv'
    elif condition == 'mvlc':
        exp_param_file = exp_param_directory + 'medium_volatility_low_conflict.csv'
    elif condition == 'hvlc':
        exp_param_file = exp_param_directory + 'high_volatility_low_conflict.csv'
    elif condition == 'lclv':
        exp_param_file = exp_param_directory + 'low_conflict_low_volatility.csv'
    elif condition == 'mclv':
        exp_param_file = exp_param_directory + 'med_conflict_low_volatility.csv'
    elif condition == 'hclv':
        exp_param_file = exp_param_directory + 'high_conflict_low_volatility.csv'
    elif condition == 'hvmc':
        exp_param_file = exp_param_directory + 'high_volatility_medium_conflict.csv'
    elif condition == 'hvhc':
        exp_param_file = exp_param_directory + 'high_volatility_high_conflict.csv'
    elif condition == 'hcmv':
        exp_param_file = exp_param_directory + 'high_conflict_medium_volatility.csv'

    else:
        sys.exit("Unknown condition.")

file_name = subj_id + "_cond" + condition + "_session" + session
data_path = os.getcwd() + '/data/' + file_name + ".csv"

if not testing and os.path.exists(data_path):
    sys.exit(file_name + " already exists!")


instructions = ("In this task, you will have a choice between two targets. When you choose one of these targets, you will lose or win points. Choosing the same target will not always give you the same points, but one is better than the other. After making your choice, you will receive feedback about the outcome. Your goal is to choose the target that gives the greatest reward. "
+"\n\nChoose the left target by pressing the left arrow key and choose the right target by pressing the right arrow key. Press any key when you're ready to begin.")
slow_trial = ("Too slow! \nChoose quickly.")
fast_trial = ("Too fast! \nSlow down.")

"""initialize dependent variables"""
rt_list = []
choice_list = []
accuracy_list = []

"""instantiate psychopy object instances"""
clock = core.Clock()
expTime_clock = core.Clock()

mbp_monitor = monitors.Monitor('mbp_15_inch')
mbp_monitor.setSizePix = [1440,900]
mbp_monitor.saveMon()

testing_monitor = monitors.Monitor('testing_computer')
testing_monitor.setSizePix = [1920,1080]
testing_monitor.saveMon()

screen_size = testing_monitor.setSizePix
# screen_size=[800,600]
center=[0,0]

# if screen_size != mbp_monitor.setSizePix:
#     center[0] = (mbp_monitor.setSizePix[0]/2) - (screen_size[0]/2)
#     center[1] = (mbp_monitor.setSizePix[1]/2) - (screen_size[1]/2)

if screen_size != testing_monitor.setSizePix:
    center[0] = (testing_monitor.setSizePix[0]/2) - (screen_size[0]/2)
    center[1] = (testing_monitor.setSizePix[1]/2) - (screen_size[1]/2)

window = visual.Window(size = screen_size, units='pix', monitor = mbp_monitor, color = [-1,-1,-1], \
       colorSpace = 'rgb', blendMode = 'avg', useFBO = True, allowGUI = \
       False,fullscr=True, pos=center)

inst_msg = visual.TextStim(win=window, units='pix',antialias='False', text=instructions, wrapWidth=screen_size[0]-400, height=screen_size[1]/25)
speed_msg = visual.TextStim(win=window, units='pix',antialias='False', text=slow_trial, pos = [0,screen_size[1]/3], height=screen_size[0]/50,
alignHoriz='center', wrapWidth=screen_size[1]*2, colorSpace='rgb',color=[1,0,0], bold=True)
choice_emphasis = visual.Rect(win=window, units='pix', height = screen_size[0]/7, width= screen_size[0]/7, lineColorSpace='rgb',lineColor=[1,1,1], lineWidth=5)
cue_1 = visual.ImageStim(window, image='./images/blue_block.png',units='pix',size=[screen_size[0]/15])
cue_0 = visual.ImageStim(window, image='./images/orange_block.png',units='pix',size=[screen_size[0]/15])
coin = visual.ImageStim(window, image='./images/coin.png',units='pix',size=[screen_size[0]/20], pos=[120,200])
treasure_chest = visual.ImageStim(window, image='./images/treasure_chest.png',units='pix',size=[screen_size[0]/18], pos=[800,screen_size[1]/2.5])

rewardMsg = visual.TextStim(win=window,units='pix',antialias='False',pos=[-20,200], colorSpace='rgb', color=[1,1,1],height=screen_size[0]/30)
totalMsg = visual.TextStim(win=window,units='pix',antialias='False',pos=[treasure_chest.pos[0]-150,treasure_chest.pos[1]], colorSpace='rgb', color=[.3,.3,.3],height=screen_size[0]/40)

cue_list = [cue_1, cue_0]

"""specify constants"""
exp_param = pd.read_csv(exp_param_file, header=0)
exp_param.columns = ['t1_r', 't2_r', 'ssd', 'cp', 'r_diff', 'mu_r_diff', 'sigma']
reward_t1 = np.round(exp_param.t1_r,2)
reward_t2 = np.round(exp_param.t2_r,2)
rewards = 10*np.transpose(np.array([reward_t1, reward_t2]))
rewards = rewards.astype(np.int)
max_reward_idx = np.argmax(rewards,1)
min_reward_idx = np.argmin(rewards,1)
n_trials = len(exp_param.cp)
n_test_trials = 6
if testing:
    n_trials = n_test_trials

"""define target coordinates"""
left_pos_x = -screen_size[0]/5
right_pos_x = screen_size[0]/5
y = 0

left_pos = [left_pos_x,y]
right_pos = [right_pos_x,y]

l_x = np.tile(left_pos_x, n_trials/2)
r_x = np.tile(right_pos_x, n_trials/2)
l_r_x_arr = np.concatenate((l_x, r_x))

"""shuffle target coordinates"""
np.random.seed()
np.random.shuffle(l_r_x_arr)

"""set constants"""
fb_time = .7
hog_period = .05
iti_min = .2
iti_max = .5
rt_max = .7
rt_min = .1

"""initalize lists"""
iti_list = []
received_rewards = []
total_rewards = []
correct_choices = []
cp_list = exp_param.cp.values[0:n_trials].tolist()

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
total_reward = 0
totalMsg.text = str(total_reward)
totalMsg.setAutoDraw(True)
treasure_chest.setAutoDraw(True)


""""present choices"""
for t in range(0,n_trials):

    cue_0.setPos([l_r_x_arr[t], y])
    cue_1.setPos([-l_r_x_arr[t], y])

    cue_list[0].setAutoDraw(True)
    cue_list[1].setAutoDraw(True)
    window.flip()

    clock.reset()
    core.wait(hog_period,hogCPUperiod=hog_period)
    data = event.waitKeys(keyList=['left','right', 'escape'],timeStamped=clock)

    choice=data[0][0]
    rt=data[0][1] - hog_period

    if rt >= rt_max:
        speed_msg.text = slow_trial
        speed_msg.draw()
    elif rt <= rt_min:
        speed_msg.text = fast_trial
        speed_msg.draw()

    """reverse high value target at changepoint"""
    if cp_list[t] == 1:
        cue_list.reverse()

    if cue_list[0].pos[0] == left_pos_x:
        correct_choice = 'left'
        correct_choices.append(0)
    else:
        correct_choice = 'right'
        correct_choices.append(1)


    if choice == 'left':
        choice_list.append(0)
        choice_emphasis.setPos(left_pos)
        rewardMsg.setPos([left_pos[0]-10, left_pos[1]+200])
        coin.setPos([left_pos[0]+110, left_pos[1]+200])

    elif choice == 'right':
        choice_list.append(1)
        choice_emphasis.setPos(right_pos)
        rewardMsg.setPos([right_pos[0]-10, right_pos[1]+200])
        coin.setPos([right_pos[0]+110, right_pos[1]+200])

    if choice == correct_choice:
        received_rewards.append(rewards[t,max_reward_idx[t]])
        rewardMsg.text = str('+' + str(rewards[t,max_reward_idx[t]]))
        total_reward += rewards[t,max_reward_idx[t]]

    elif choice != correct_choice:
        received_rewards.append(rewards[t,min_reward_idx[t]])
        rewardMsg.text = str('+' + str(rewards[t,min_reward_idx[t]]))
        total_reward += rewards[t,min_reward_idx[t]]

    total_rewards.append(total_reward)
    rt_list.append(rt)

    totalMsg.text = str(total_reward)

    if choice == 'escape':
        sys.exit('escape key pressed.')

    choice_emphasis.draw()
    rewardMsg.draw()
    totalMsg.draw()
    coin.draw()
    window.flip()
    core.wait(fb_time)

    cue_list[0].setAutoDraw(False)
    cue_list[1].setAutoDraw(False)
    window.flip()
    clock.reset()

    """jitter iti"""
    iti = random.uniform(iti_min, iti_max)
    iti_list.append(iti)
    core.wait(iti)

    accuracy_list.append(choice == correct_choice)

total_exp_time=np.tile(expTime_clock.getTime(),n_trials)
"""save data"""
header = ("choice, rt, cp, accuracy, reward, cumulative_reward, solution, total_exp_time")
data = np.transpose(np.array([choice_list, rt_list, cp_list, accuracy_list,
 received_rewards, total_rewards, correct_choices,total_exp_time]))

np.savetxt(data_path, data, header=header, delimiter=',',comments='')

window.close()
