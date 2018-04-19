import os
import numpy as np
from itertools import permutations

def counterbalance_conditions(condition_list, condition_path=os.getcwd(), filename='/test_cb_conds.csv'):

    n_conditions = len(condition_list)

    #permute condition order
    permuted_order = list(permutations(condition_list))

    #find n_subjects needed for full permutation
    n_subjects = len(permuted_order)

    trial_file_number = np.arange(0,n_subjects)
    coax_id = np.full(n_subjects,np.nan)

    #automatically construct order header according to number of conditions
    order_base_txt = np.repeat('condition', n_conditions)
    order_n = list(map(str, np.arange(0, n_conditions)))
    order = [order_base_txt[i]+order_n[i] for i in range(n_conditions)]

    #get rid of brackets & convert list to str
    order_header = str(order)[1:-1]

    #so that the first n subjects don't have the same beginning condition
    #eliminates need to shuffle sub. number
    #only shuffles first axis (rows) so column order is maintained
    np.random.shuffle(permuted_order)


    header = ("coax_id, trial_file_set_number, "+ order_header)
    write_data = np.column_stack((coax_id, trial_file_number, permuted_order))
    np.savetxt(condition_path+filename, write_data, header=header, delimiter=',',comments='',fmt='%s')
