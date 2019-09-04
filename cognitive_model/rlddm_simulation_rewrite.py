import numpy as np
import pandas as pd
import numba as nb

from numba.decorators import jit
from np.random import random_sample
from numba import float64, int64, vectorize, boolean
from Equation import Expression

from bayesian_belief_model_p import update_bayesian_belief


def sim_ddm_trials(exp_constants, initial_decision_parameters,
learning_algorithm, behavior):
    """ input the parameters needed for
    reinforcement learning and drift-diffusion modeling simulation.
    wraps around the subsequent functions.

    :: Arguments ::
    exp_parameters (pd.dataframe): containing reward for each choice
    and change point indicator (n_trials x 2)


    :: Returns ::
    sim_behavior (pd.DataFrame): simulated rt and choice data
    (n_trials x 2)
    sim_params (pd.DataFrame): simulated decision parameters,
    dynamically updated (n_trials x n_parameters)
    traces (np.ndarray): evolution of the evidence trace over
    time steps (n_trials x n_time)
"""

    for t in trials:
        choices[t], rts[t] = adapt_ddm()
        learning_signals[t] = update_pseudobayesian_belief()


    return choices, rts, learning_signals



def define_learning_algorithm(*decision_parameters, *learning_signals,
 *learning_rates):
"""
    :: Arguments ::
    decision_parameters (dict):
        a: decision threshold
        v: drift-rate
        t: non-decision time

    learning_signals (dict):
        cpp: change-point probability
        B: signed belief in the value of the optimal target
        learning_rates (dict): the learning rates for each component of the
        learning algorithm (0,1)


    :: Returns ::
    learning_algorithm (dict): the learning algorithm, as defined by the combination of
    learning signals and decision parameters


"""
    # specify all rational combinations of learning signals and decision processes

    a, v, t, z = decision_parameters
    cpp, B = learning_signals


    # + AND - product(decision_param, learning_signals) - irrational comb.




    # TODO: this is a bad idea... pr. relies on eval. use product.
    # model_input = input('Enter learning algorithm spec. (dp, lr, learning signal): ')
    # model = Expression(model_input)

    # output = model(decision_p=decision_p, learning_rate=learning_rate,
    #  learning_signal=learning_signal)

    # if a in decision_parameters and cpp in learning_signals:
    #     a_model = 'a_t+1 = a - beta*cpp'
    #     model_list.append(a_model)
    #
    # elif v in decision_parameters and B in learning_signals:
    #     v_model = 'v_t+1 = v + beta*B'
    #     model_list.append(v_model)

    return parameters


def initialize_ddm_params(**args, si=.1, tr=.25,
 dt=.001, timebound=.7):
    """ define the initial values for the ddm parameters.

    :: Arguments ::
        initial_decision_parameters (dict of variable size):
            a: decision threshold (0,1)
            v: drift-rate

            tr: non-decision time
            si: diffusion constant (noise)
            dt: time step
            timebound: maximum rt in s


    :: Returns ::
        returns initialized ddm parameters (list):
            z: unbiased starting point (a/2)
            dx: step size for the vertical movement of the evidence trace,
            derived from diffusion constant and the time step
            a
            v
            tr
            si
            dx
            dt
            evidence: initialized evidence time series within a trial


    """


    z = a * .5 # unbiased starting point
    dx = si * np.sqrt(dt)
    n_timesteps = int(timebound // dt) # number of time steps

    intialized_ddm_params = [a, v, z, tr, si, dx, dt, timebound, n_timesteps] # scalars ONLY
    evidence = np.full((n_timesteps, n_trials), np.nan)


    return intialized_ddm_params, evidence


def get_exp_constants(exp_parameters):

    n_trials = len(exp_parameters)

    reward_targets = np.column_stack((exp_parameters.reward_t0, exp_parameters.reward_t1))

    n_choices = reward_targets.ndim

    H = np.sum(exp_parameters.cp == 1) / n_trials # hazard rate

    lower_r, upper_r = reward_targets.min, reward_targets.max # lower and upper bound for reward value

    high = upper_r - lower_r # reward value range

    sN = exp_parameters.sd # standard dev. of reward distribution

    constants = [n_trials, n_choices, H, lower_r, upper_r, high]


    return constants

def initalize_behavior(*constants):

    rts, choices = np.full(n_trials, np.nan), np.full(n_trials, np.nan)

    return rts, choices


def initialize_pseudobayesian_estimates(exp_parameters):

# TODO: update these to reflect inputs

     # H =  expParam[( expParam.cp == 1)].shape[0] /  expParam.shape[0]
     # sN = np.ones_like( reaction_times)
     # low =  reward_targets.min()
     # up =  reward_targets.max()
     # high =  up -  low
     # nChoices =  reward_targets.shape[1]


     B = np.zeros([ nTrials,  nChoices])
     lr = np.zeros([ nTrials])
     signed_B_diff = np.zeros_like( lr)
     B_diff = np.zeros_like( lr)
     rpe = np.zeros_like( B)
     CPP = np.zeros_like( lr)
     MC = np.zeros_like( lr) + 0.5
     epoch_length = np.zeros_like( lr) + 1
     sF = np.zeros_like( lr)


def adapt_ddm(decision_parameters, evidence, random_probability):

    v_prob = .5 * (1 + (v * np.sqrt()dt)/si)
    evidence[0] = z_init

    for timestep in range(n_timesteps):

        if random_probability[timestep] < v_prob:
            evidence += dx
        else:
            evidence -= dx

        if evidence >= a_upper_bound:
            rt = tr + (n_timesteps * dt)
            choice = 1

            return rt, choice

        elif evidence <= a_lower_bound:
            rt = tr + (n_timesteps * dt)
            choice = 0

            return rt, choice

    return np.nan, np.nan # if the boundary was never crossed, return nans for rt and choice


def update_pseudobayesian_belief(reward, choices,
learning_signals, decision_parameters, learning_algorithm, trial):

    learning_signals = update_bayesian_belief(reward, choices, learning_signals)

    if t < n_trials - 1:

        # TODO: add flexible model spec.

        if

            v[t+1] = v + learning_rates['alpha'] * signed_B_diff
            a[t+1] = a_init - learning_rates['beta'] * cpp




        if v[t+1] > v_max:
            v[t+1] = v_max * np.sign(v[t+1])

    # TODO: pack learning signals into array
    return learning_signals

def calculate_accuracy(choices, correct_choices):

    """ determine whether the most probably rewarding target was selected.

        :: Arguments ::
        choices (n_trials 1d array): selected choices (0 | 1)
        correct_choices (n_trials 1d array): the most probably rewarding choices

     """

    correct_targets = np.argmax(correct_choices,1)
    choice_acc = correct_targets == choices

    return choice_acc
