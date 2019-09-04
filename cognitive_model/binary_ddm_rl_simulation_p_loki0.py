import os

from simulation_functions_loki0 import Simulation
import numpy as np
from multiprocessing import Pool
from pickle_objects import save_object

#from time import time


#alpha: belief lr
#beta: cpp lr
#delta: z lr


a_baseline = 0.5


#old values
mod_alpha = 2
mod_beta = a_baseline / 1.5
mod_delta = 0
mod_zeta = 0


mod_learning_rates = {'alpha': mod_alpha,'beta': mod_beta, 'delta': mod_delta, 'zeta': mod_zeta}


#n_subjects = 50
n_subjects = 24
path = os.getcwd() +'/simulation_reward_prob/mod/'


#a = .10 # boundary height
#v = .14 # strong drift-rate
#tr = .25 # nondecision time (in seconds)
#z = .5 # starting point ([0,1], fraction of a)



drift_start, bound_start, sp_start, tr_start = (0.14,0.1,0.5,0.25)


def cpu_simulation(model,mod_learning_rates, drift_start, bound_start, sp_start, tr_start, pathstring):
    sim = Simulation(model,mod_learning_rates,drift_start, bound_start, sp_start, tr_start, pathstring)
    sim.adapt_ddm()
    return sim



"""hypothesized update for bound + drift"""

sim_hv_comb_args = [(3,mod_learning_rates,drift_start,bound_start, sp_start, tr_start, path+'hv_'+str(s)+'.csv') for s in range(n_subjects) ]
sim_lv_comb_args = [(3,mod_learning_rates,drift_start,bound_start, sp_start, tr_start, path+'lv_'+str(s)+'.csv') for s in range(n_subjects) ]
sim_hc_comb_args = [(3,mod_learning_rates,drift_start,bound_start, sp_start, tr_start, path+'hc_'+str(s)+'.csv') for s in range(n_subjects) ]
sim_lc_comb_args = [(3,mod_learning_rates,drift_start,bound_start, sp_start, tr_start, path+'lc_'+str(s)+'.csv') for s in range(n_subjects) ]

with Pool() as p:
    sim_hv_comb=p.starmap(cpu_simulation,sim_hv_comb_args)

    sim_lv_comb=p.starmap(cpu_simulation,sim_lv_comb_args)

    sim_hc_comb=p.starmap(cpu_simulation,sim_hc_comb_args)

    sim_lc_comb=p.starmap(cpu_simulation,sim_lc_comb_args)




all_models_dict = {'sim_hv_comb': sim_hv_comb, 'sim_lv_comb': sim_lv_comb,
                   'sim_hc_comb': sim_hc_comb, 'sim_lc_comb': sim_lc_comb}


for sim, iteration in zip(sim_hv_comb, range(len(sim_hv_comb))):
    sim.condition = 'hv'
    sim.sim_n = iteration
for sim, iteration in zip(sim_lv_comb, range(len(sim_lv_comb))):
    sim.condition = 'lv'
    sim.sim_n = iteration


for sim, iteration in zip(sim_hc_comb, range(len(sim_hc_comb))):
    sim.condition = 'hc'
    sim.sim_n = iteration

for sim, iteration in zip(sim_lc_comb, range(len(sim_lc_comb))):
    sim.condition = 'lc'
    sim.sim_n = iteration



all_models_list = sim_hv_comb + sim_lv_comb + sim_hc_comb + sim_lc_comb



home_path = os.path.expanduser('~')
sim_data_path = home_path + '/Dropbox/loki_0/spydr_scripts/sim_obj_pkl/'


#save each simulation as a pickled object
[save_object(sim, sim_data_path+'sim' + str(sim.sim_n) + '_'+ sim.condition + '.pkl') for sim in all_models_list]
