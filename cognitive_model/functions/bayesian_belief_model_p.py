
# coding: utf-8

from scipy.stats.distributions import uniform, norm
import random
import numpy as np

def update_bayesian_belief(t, nTrials, prob_reward_targets, H, sN, low, up, high,
choices, B, signed_B_diff, B_diff, lr, rpe, CPP, MC, epoch_length, sF):

    unifpdf = lambda x_t: uniform(low, high).pdf(x_t)
    normpdf = lambda mu, sig, x_t: norm(mu, sig).pdf(x_t)

    choice = choices[t]
    if  choice == 0:
         nonchoice = 1
    elif choice == 1:
         nonchoice = 0

    # r_diff = prob_reward_targets[:,1]-prob_reward_targets[:,0]

    # rpe[ t, choice] =  prob_reward_targets[t,  choice] -  B[  t,  choice]
    # sF[ t] = np.sqrt( sN[t]**2 + ( sN[t]**2 * (1- MC[ t]))/( MC[ t]))
    # u_val = unifpdf(prob_reward_targets[t,choice])
    # n_val = normpdf(B[t,choice], sF[ t], prob_reward_targets[t,choice])
    # # n_val = normpdf( B[ t,choice], sF[ t], B[t-1,choice]) #could also be last time it chose this one
    # # n_val = normpdf(signed_B_diff[t], sF[ t], r_diff[t])
    # # CPP[ t] = (u_val*H)/((u_val*H) + (n_val*(1-H)))
    # CPP[ t] = (u_val*H)/((u_val*H) + (n_val*(1-H)))
    # lr[ t] =  CPP[ t] + (1- MC[ t])*(1- CPP[ t])

    rpe[t, choice] =  prob_reward_targets[t,  choice] -  B[t,  choice]
    # rpe[t] =  prob_reward_targets[t,  choice] -  B[ t,choice]
    sF[ t] = np.sqrt( sN[t]**2 + ( sN[t]**2 * (1- MC[ t]))/( MC[ t]))
    u_val = unifpdf(prob_reward_targets[t,choice])
    n_val = normpdf( B[ t,choice], sF[ t], prob_reward_targets[t,choice])
    CPP[ t] = (u_val*H)/((u_val*H) + (n_val*(1-H)))
    lr[ t] =  CPP[ t] + (1- MC[ t])*(1- CPP[ t])

#     stable_cpp[t] = 0.33*stable_cpp  + CPP[t] #cpp is not predictive (reactive).
#need to edit this rule.


     # Next trial calculations
    if t < (nTrials-1):
        B[ t+1, choice] =  B[ t, choice] + lr[ t]*rpe[ t,choice]
        #reset belief in difference to 0.5 whenever there actually is a cp
        #this only makes sense if the feedback is cumulative_reward...
        # B[ t+1,  nonchoice] =  B[ t, nonchoice]*(1-CPP[t])+CPP[t]*.5

        #instead, for values, decay to average val for both targets ((3+0)/2)
        B[ t+1,  nonchoice] =  B[ t, nonchoice]*(1-CPP[t])+CPP[t]*1.5

        # B[ t+1,  nonchoice] =  1-B[ t, choice]
        # B[ t+1,  nonchoice] =  B[ t, nonchoice]*(1-CPP[t])
        # B[ t+1,  nonchoice] =  B[ t, nonchoice]


        # B[ t+1,  nonchoice] =  B[ t, nonchoice]*(1-lr[ t]) #an alternative, but decays quickly
        signed_B_diff[ t+1] = B[ t,1] - B[ t,0]
        B_diff[ t+1] = B[ t,choice] - B[ t,nonchoice]
        # print(signed_B_diff)

        # B_diff[ t+1] =  B[ t, choice] - B[ t, nonchoice]
        # term1 =  CPP[ t]* sN[t]**2
        # term2 = (1- CPP[ t])*(1- MC[ t])* sN[t]**2
        # term3 =  CPP[ t]*(1- CPP[ t])*(rpe[ t,choice]*( MC[ t]))**2
        # MC[ t+1] = 1 - ((term1+term2+term3)/(term1+term2+term3 +  sN[t]**2))
        # epoch_length[ t+1] = (epoch_length[ t] + 1)*(1- CPP[ t]) +  CPP[ t]
        term1 =  CPP[ t]* sN[t]**2
        term2 = (1- CPP[ t])*(1- MC[ t])* sN[t]**2
        term3 =  CPP[ t]*(1- CPP[ t])*(rpe[ t,choice]*( MC[ t]))**2
        MC[ t+1] = 1 - ((term1+term2+term3)/(term1+term2+term3 +  sN[t]**2))
        epoch_length[ t+1] = (epoch_length[ t] + 1)*(1- CPP[ t]) +  CPP[ t]

    return [B,signed_B_diff,B_diff,lr,rpe,CPP,MC,epoch_length,sF]
