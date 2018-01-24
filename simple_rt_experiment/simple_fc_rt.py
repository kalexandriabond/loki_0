
import os,random,sys
from psychopy import visual,core,event,monitors
import pandas as pd
import numpy as np
from random import shuffle

"""FIX CHANGEPOINTS. DOES NOT SWITCH IDENTITIES."""

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
    exp_param_file = exp_param_directory + 'rprobe_test.csv'
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

"""instantiate psychopy object instances"""
clock = core.Clock()

mbp_monitor = monitors.Monitor('mbp_15_inch')
mbp_monitor.setSizePix = [1920,1080]
mbp_monitor.saveMon()

testing_monitor = monitors.Monitor('testing_computer')
testing_monitor.setSizePix = [1000,800]
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
orange_block = visual.ImageStim(window, image='./images/orange_block.png',units='pix',size=[screen_size[0]/10])
blue_block = visual.ImageStim(window, image='./images/blue_block.png',units='pix',size=[screen_size[0]/10])
coin = visual.ImageStim(window, image='./images/coin.png',units='pix',size=[screen_size[0]/20], pos=[120,200])
treasure_chest = visual.ImageStim(window, image='./images/treasure_chest.png',units='pix',size=[screen_size[0]/18], pos=[800,screen_size[1]/2.5])

rewardMsg = visual.TextStim(win=window,units='pix',antialias='False',pos=[-20,200], colorSpace='rgb', color=[1,1,1],height=screen_size[0]/30)
totalMsg = visual.TextStim(win=window,units='pix',antialias='False',pos=[treasure_chest.pos[0]-150,treasure_chest.pos[1]], colorSpace='rgb', color=[.3,.3,.3],height=screen_size[0]/40)

"""specify constants"""
exp_param = pd.read_csv(exp_param_file, header=0)
exp_param.columns = ['t1_r', 't2_r', 'ssd', 'cp', 'r_diff', 'mu_r_diff', 'sigma']
reward_t1 = np.round(exp_param.t1_r,2)
reward_t2 = np.round(exp_param.t2_r,2)
rewards = 100*np.transpose(np.array([reward_t1, reward_t2]))
rewards = rewards.astype(np.int)
max_reward_idx = np.argmax(rewards,1)
min_reward_idx = np.argmin(rewards,1)



max_rt = .7
min_rt = .1
left_pos_x = [screen_size[0]/5]
right_pos_x = [-screen_size[0]/5]
l_r_y = 0

left_pos = [-screen_size[0]/5, 0]
right_pos = [screen_size[0]/5, 0]

n_trials = len(exp_param.cp)
n_test_trials = 25

l_x = np.tile(left_pos_x, n_trials/2)
r_x = np.tile(right_pos_x, n_trials/2)
l_r_x_arr = np.concatenate((l_x, r_x))
shuffle(l_r_x_arr)

fb_time = .7


iti_min = .1
iti_max = .5
iti_list = []

if testing:
    n_trials = n_test_trials

cp_list = exp_param.cp.values[0:n_trials].tolist()

correct_left_transform_idx = np.where(l_r_x_arr == left_pos_x)

solutions = np.ones((len(reward_t1)))
solutions[correct_left_transform_idx] = 0
print(solutions)
"""define forced choice trial type"""
def ForcedChoiceTrial(max_reward_idx,t):

    orange_block.setPos([l_r_x_arr[t], l_r_y])
    blue_block.setPos([-l_r_x_arr[t], l_r_y])

    orange_block.setAutoDraw(True)
    blue_block.setAutoDraw(True)

    iti = random.uniform(iti_min, iti_max)
    iti_list.append(iti)
    core.wait(iti)

    window.flip()
    clock.reset()
    # data = event.waitKeys(keyList=['left','right','escape'],timeStamped=clock)
    data = event.waitKeys(keyList=['left','right', 'escape'],timeStamped=True)

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
total_reward = 0
totalMsg.text = str(total_reward)
totalMsg.setAutoDraw(True)
treasure_chest.setAutoDraw(True)


""""present choices"""
for t in range(0,n_trials):
    orange_block.setAutoDraw(False)
    blue_block.setAutoDraw(False)
    window.flip()

    choice,rt = ForcedChoiceTrial(max_reward_idx[t],t)
    rt_list.append(rt)


    if choice == 'left':
        choice = 0
        choice_emphasis.setPos(left_pos)
        rewardMsg.setPos([left_pos[0]-10, left_pos[1]+200])
        coin.setPos([left_pos[0]+110, left_pos[1]+200])

    elif choice == 'right':
        choice = 1
        choice_emphasis.setPos(right_pos)
        rewardMsg.setPos([right_pos[0]-10, right_pos[1]+200])
        coin.setPos([right_pos[0]+110, right_pos[1]+200])

    choice_list.append(choice)


    if choice == solutions[t]:
        rewardMsg.text = str('+' + str(rewards[t,max_reward_idx[t]]))
        total_reward += rewards[t,max_reward_idx[t]]
    elif choice != solutions[t]:
        rewardMsg.text = str('+' + str(rewards[t,min_reward_idx[t]]))
        total_reward += rewards[t,max_reward_idx[t]]

    accuracy_list = np.equal(solutions[:t+1], np.asarray(choice_list[:t+1]))


    totalMsg.text = str(total_reward)

    if choice == 'escape':
        sys.exit('escape key pressed.')

    choice_emphasis.draw()
    rewardMsg.draw()
    totalMsg.draw()
    coin.draw()
    window.flip()
    core.wait(fb_time)



"""save data"""
header = ("choice, rt, cp, accuracy")
dvs = np.transpose(np.array([choice_list, rt_list, cp_list, accuracy_list]))
np.savetxt(data_path, dvs, header = header, delimiter=',',comments='')

window.close()
