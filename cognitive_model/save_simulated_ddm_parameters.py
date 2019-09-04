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
simulated_ddm_parameter_dfs = [pd.DataFrame(dict((key, value) for (key, value) in zip(['a','v', 'tr', 'z', 'condition', 'sim_n', 'accuracy'],
                      [sim.a_current, sim.v_diff_current, sim.tr, sim.z_current, sim.condition, sim.sim_n, sim.choiceAcc]))) for sim in av_model_sims]



simulated_ddm_parameter_df = pd.concat(simulated_ddm_parameter_dfs)


simulated_ddm_parameter_df.to_csv(sim_data_path + 'simulated_ddm_parameters.csv', index=False)
