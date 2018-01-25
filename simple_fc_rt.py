
import os,sys,random
from psychopy import prefs
prefs.general['audioLib'] = ['pygame']
from psychopy import visual,core,event,monitors,sound,info
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
    exp_param_file = exp_param_directory + 'hv_hc0.csv'
else:
    subj_id = raw_input("Subject ID: ")
    condition = raw_input("Condition: ")
    session = raw_input("Session: ")

    if condition == 'lvlc':
        exp_param_file = exp_param_directory + 'lv_lc' + session + '.csv'
    elif condition == 'hvlc':
        exp_param_file = exp_param_directory + 'hv_lc' + session + '.csv'
    elif condition == 'lvlc':
        exp_param_file = exp_param_directory + 'lv_lc'  + session + '.csv'
    elif condition == 'lvhc':
        exp_param_file = exp_param_directory + 'lv_hc' + session + '.csv'
    elif condition == 'hvhc':
        exp_param_file = exp_param_directory + 'hv_hc' + session + '.csv'


    else:
        sys.exit("Unknown condition.")

file_name = subj_id + condition + "_session" + session
data_path = os.getcwd() + '/data/' + file_name + ".csv"
run_info_path = os.getcwd() + '/data/' + file_name + "_runInfo.csv"


if not testing and os.path.exists(data_path):
    sys.exit(file_name + " already exists!")


instructions = ("In this task, you will be able to choose between opening one of two boxes. When you open a box, you will get a certain number of coins. However, opening the same box will not always give you the same number of coins. After making your choice, you will receive feedback about how much money you have. Your goal is to make as much money as possible. "
+"\n\nChoose the left box by pressing the left arrow key and choose the right box by pressing the right arrow key. Press any key when you're ready to begin.")
slow_trial = ("Too slow! \nChoose quickly.")
# fast_trial = ("Too fast! \nSlow down.")
break_inst = ("Feel free to take a break! \nWhen you're ready, press any key to continue.")


"""initialize dependent variables"""
rt_list = []
choice_list = []
accuracy_list = []

"""instantiate psychopy object instances"""
clock = core.Clock()
expTime_clock = core.Clock()
trialTime_clock = core.Clock()

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

break_msg = visual.TextStim(win=window, units='pix',antialias='False', text=break_inst, wrapWidth=screen_size[0]-400, height=screen_size[1]/25)
inst_msg = visual.TextStim(win=window, units='pix',antialias='False', text=instructions, wrapWidth=screen_size[0]-400, height=screen_size[1]/25)
end_msg = visual.TextStim(win=window, units='pix', antialias='False', wrapWidth=screen_size[0]-400, height=screen_size[1]/25)
speed_msg = visual.TextStim(win=window, units='pix',antialias='False', text=slow_trial,  wrapWidth=screen_size[0]-400, height=screen_size[1]/15,
alignHoriz='center', colorSpace='rgb',color=[1,0,0], bold=True)


choice_emphasis = visual.Rect(win=window, units='pix', height = screen_size[0]/7, width= screen_size[0]/7, lineColorSpace='rgb',lineColor=[1,1,1], lineWidth=5)
cue_1 = visual.ImageStim(window, image='./images/blue_block.png',units='pix',size=[screen_size[0]/15])
cue_0 = visual.ImageStim(window, image='./images/orange_block.png',units='pix',size=[screen_size[0]/15])
coin = visual.ImageStim(window, image='./images/coin.png',units='pix',size=[screen_size[0]/20], pos=[120,200])
treasure_chest = visual.ImageStim(window, image='./images/treasure_chest.png',units='pix',size=[screen_size[0]/18], pos=[800,screen_size[1]/2.5])

beep = sound.Sound('./sounds/buzz.wav')
coin_sound = sound.Sound('./sounds/add_coin.wav')
break_block = sound.Sound('./sounds/break_block.wav')

runtimeInfo = info.RunTimeInfo(author='kb',win=window,userProcsDetailed=False, verbose=True)
rewardMsg = visual.TextStim(win=window,units='pix',antialias='False',pos=[-20,200], colorSpace='rgb', color=[1,1,1],height=screen_size[0]/30)
totalMsg = visual.TextStim(win=window,units='pix',antialias='False',pos=[treasure_chest.pos[0]-150,treasure_chest.pos[1]], colorSpace='rgb', color=[.3,.3,.3],height=screen_size[0]/40)

cue_list = [cue_1, cue_0]

"""specify constants"""
exp_param = pd.read_csv(exp_param_file, header=0)
exp_param.columns = ['t1_r', 't2_r', 'ssd', 'cp', 'r_diff', 'mu_r_diff', 'sigma']
reward_t1 = np.round(exp_param.t1_r,2)
reward_t2 = np.round(exp_param.t2_r,2)
rewards = 100*np.transpose(np.array([reward_t1, reward_t2]))
rewards = rewards.astype('int')
max_reward_idx = np.argmax(rewards,1)
min_reward_idx = np.argmin(rewards,1)
n_trials = len(exp_param.cp)
n_test_trials = 2
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
fb_time = .9
iti_min = .25
iti_max = .75
rt_max = 1
rt_min = .1
left_key = 'f'
right_key = 'd'
escape_key = 'e'

"""initalize lists"""
iti_list = []
received_rewards = []
total_rewards = []
correct_choices = []
trial_time = []
cp_with_slow = []
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
t=0

totalMsg.text = str(total_reward)
totalMsg.setAutoDraw(True)
treasure_chest.setAutoDraw(True)


""""present choices"""
while t < n_trials:
    if t == int(n_trials/2):
        if cp_list[t+1] != 1:
            print("ERROR! slice is incorrect.")
        break_msg.setAutoDraw(True)
        window.flip()
        clock.reset()
        break_data = event.waitKeys(timeStamped=clock)
        break_time = break_data[0][1]
        break_msg.setAutoDraw(False)

    cue_0.setPos([l_r_x_arr[t], y])
    cue_1.setPos([-l_r_x_arr[t], y])

    cue_list[0].setAutoDraw(True)
    cue_list[1].setAutoDraw(True)
    window.flip()

    clock.reset()
    data = event.waitKeys(keyList=[left_key,right_key, 'escape'])
    rt=clock.getTime()
    choice=data[0][0]
    if choice==escape_key:
        sys.exit('escape key pressed.')

    """reverse high value target at changepoint"""
    if cp_list[t] == 1:
        cue_list.reverse()

    if cue_list[0].pos[0] == left_pos_x:
        correct_choice = left_key
        correct_choices.append(0)
    else:
        correct_choice = right_key
        correct_choices.append(1)


    if choice == left_key:
        choice_list.append(0)
        choice_emphasis.setPos(left_pos)
        rewardMsg.setPos([left_pos[0]-10, left_pos[1]+200])
        coin.setPos([left_pos[0]+110, left_pos[1]+200])

    elif choice == right_key:
        choice_list.append(1)
        choice_emphasis.setPos(right_pos)
        rewardMsg.setPos([right_pos[0]-10, right_pos[1]+200])
        coin.setPos([right_pos[0]+110, right_pos[1]+200])
    if rt < rt_max:
        if choice == correct_choice:
            received_rewards.append(rewards[t,max_reward_idx[t]])
            rewardMsg.text = str('+' + str(rewards[t,max_reward_idx[t]]))
            total_reward += rewards[t,max_reward_idx[t]]
            core.wait(0.1)
            coin_sound.play()

        elif choice != correct_choice:
            received_rewards.append(rewards[t,min_reward_idx[t]])
            rewardMsg.text = str('+' + str(rewards[t,min_reward_idx[t]]))
            total_reward += rewards[t,min_reward_idx[t]]
            coin_sound.play()

        totalMsg.text = str("{:,}".format(total_reward))

        choice_emphasis.draw()
        rewardMsg.draw()
        totalMsg.draw()
        coin.draw()
        window.flip()
        core.wait(fb_time)


    if rt >= rt_max:
        window.flip()
        speed_msg.text = slow_trial
        beep.play()
        cue_list[0].setAutoDraw(False)
        cue_list[1].setAutoDraw(False)
        speed_msg.setAutoDraw(True)
        window.flip()
        core.wait(fb_time)
        speed_msg.setAutoDraw(False)
        received_rewards.append(0)
        cp_with_slow.append(np.nan)

    # elif rt <= rt_min: #think about this
    #     speed_msg.text = fast_trial
    #     speed_msg.draw()
    #     received_rewards.append(0)
    else:
        cp_with_slow.append(cp_list[t])


    total_rewards.append(total_reward)
    rt_list.append(rt)

    cue_list[0].setAutoDraw(False)
    cue_list[1].setAutoDraw(False)
    window.flip()
    clock.reset()

    """jitter iti"""
    iti = random.uniform(iti_min, iti_max)
    iti_list.append(iti)
    core.wait(iti)

    accuracy_list.append(choice == correct_choice)
    trial_time.append(trialTime_clock.getTime())
    trialTime_clock.reset()

    if rt > rt_max:
        continue
    else:
        t+=1

total_exp_time=expTime_clock.getTime()

"""save data"""
header = ("choice, accuracy, solution,  reward, cumulative_reward, rt,  total_trial_time, iti, cp_with_slow")
data = np.transpose(np.matrix((choice_list, accuracy_list,correct_choices,
 received_rewards, total_rewards, rt_list, trial_time, iti_list, cp_with_slow)))

runtime_data = np.matrix((str(runtimeInfo['psychopyVersion']), str(runtimeInfo['pythonVersion']),
str(runtimeInfo['pythonScipyVersion']),str(runtimeInfo['pythonPygletVersion']),
str(runtimeInfo['pythonPygameVersion']),str(runtimeInfo['pythonNumpyVersion']),str(runtimeInfo['pythonWxVersion']),
str(runtimeInfo['windowRefreshTimeAvg_ms']), str(runtimeInfo['experimentRunTime']),
str(runtimeInfo['experimentScript.directory']),str(runtimeInfo['systemRebooted']),
str(runtimeInfo['systemPlatform']),str(runtimeInfo['systemHaveInternetAccess']), total_exp_time, break_time))

runtime_header = ("psychopy_version, python_version, pythonScipyVersion,\
pyglet_version, pygame_version, numpy_version,wx_version, window_refresh_time_avg_ms,\
begin_time, exp_dir, last_sys_reboot, system_platform, internet_access,\
 total_exp_time, break_time")
np.savetxt(data_path, data, header=header, delimiter=',',comments='')
np.savetxt(run_info_path,runtime_data, header=runtime_header,delimiter=',',comments='',fmt="%s")

end_msg.text = ("Awesome! You earned " + totalMsg.text + " coins. \nLet the experimenter know that you're finished.")


"""dismiss participant"""
instruction_phase = True
while instruction_phase:
    end_msg.setAutoDraw(True)
    end_keys = event.getKeys()
    if 'escape' in end_keys:
        sys.exit('escape key pressed.')
    if len(end_keys) > 0:
        instruction_phase = False
    if event.getKeys(keyList=['escape']):
        sys.exit('escape key pressed.')
    if instruction_phase:
        window.flip()

end_msg.setAutoDraw(False)
window.flip()

window.close()
