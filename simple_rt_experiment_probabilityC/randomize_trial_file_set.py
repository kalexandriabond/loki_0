import numpy as np
import os

n_subjects = int(raw_input('n_subjects? '))
filename = raw_input('filename? ')
condition_path = os.getcwd()+'/'+filename+'.csv'

trial_file_set = np.arange(0,n_subjects)
np.random.seed()
np.random.shuffle(trial_file_set)

np.savetxt(condition_path, trial_file_set, delimiter=',',comments='',fmt='%s')
