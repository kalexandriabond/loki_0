#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:23:04 2019

@author: 67981492
"""

import os
import numpy as np
from pickle_objects import load_object
import pandas as pd



home_path = os.path.expanduser('~')
sim_data_path = home_path + '/Dropbox/loki_0/spydr_scripts/sim_obj_pkl/'


n_sims = 24 
n_conditions = 4 

av_model_sims = [load_object(sim_data_path+'sim' + str(sim.sim_n) + '_'+ sim.condition + '.pkl') for sim in all_models_list]


print('is all simulated data loaded?', len(av_model_sims) == (n_sims * n_conditions)) #check that all are loaded


#save simulated_behavior
simulated_behavior_dfs = [pd.DataFrame(dict((key, value) for (key, value) in zip(['reaction_times','accuracy', 'condition', 'sim_n'],
                      [sim.reaction_times, sim.choiceAcc, sim.condition, sim.sim_n]))) for sim in av_model_sims]


simulated_behavior_df = pd.concat(simulated_behavior_dfs)

#simulated_behavior_df['lambda_val'] = np.nan
#
#for reward_code, lambda_val in zip(reward_codes, lambda_vals): #specify lambda for each reward code
#    simulated_behavior_df.loc[simulated_behavior_df.reward_code == str(reward_code), 'lambda_val'] = lambda_val
#


simulated_behavior_df.to_csv(sim_data_path + 'simulated_behavior.csv', index=False)
