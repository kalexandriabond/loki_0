import random, os
import numpy as np


filename = raw_input('filename? ')
condition_path = os.getcwd()+'/'+filename+'.csv'

n_conditions = int(raw_input('n_conditions? '))
n_subjects = int(raw_input('n_subjects? '))
subject_list = np.arange(0,n_subjects)

order = np.arange(0,n_conditions)
c_order = np.zeros((n_subjects, n_conditions)) + np.nan
np.random.seed()
for s in np.arange(0, n_subjects):
    np.random.shuffle(order)
    c_order[s,:]=order

header = ("subject, first_cond, second_cond, third_cond, fourth_cond")
write_data = np.column_stack((subject_list,c_order))
print(write_data)
np.savetxt(condition_path, write_data, header=header, delimiter=',',comments='',fmt='%s')
