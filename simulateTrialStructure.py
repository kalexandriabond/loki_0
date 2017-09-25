
# coding: utf-8

# # volVal trial structure generation

# In[15]:


#write simulator for various trial structures


# In[16]:


import matplotlib.pyplot as plt
import numpy as np
import random as rd
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import csv
import seaborn as sns
import sys
import time
from jupyterthemes import jtplot
# jtplot.style(theme='grade3')
# sns.set()
get_ipython().magic('matplotlib inline')


# In[17]:


def genChangePoint(constantChangePoint, lambdaV, nTrials_approx, propStopTrials, ssd):

    nChangePoints = int(nTrials_approx/lambdaV)

    if np.mod(nChangePoints,2) != 0:
        nChangePoints += 1

    changePoints = []
    nTrials = 1

    #set tolerance for nTrials to keep moderately similar between subjects
    nTrialTol = .05 * nTrials_approx
    low_nTrialBound = nTrials_approx - nTrialTol
    high_nTrialBound = nTrials_approx + nTrialTol

    #sample changePoint
    while nTrials < low_nTrialBound  or nTrials > high_nTrialBound or np.mod(
                nTrials,10) != 0 or np.mod(nTrials*propStopTrials, len(ssd)) != 0:

        if constantChangePoint == 0:
            changePoints = list(np.random.poisson(lambdaV,nChangePoints))
            #convert to trial number indices
            changeIdx = list(np.cumsum(changePoints))
            nTrials = changeIdx[-1]
            changePoint_vec = np.zeros((nTrials))
            #mark change point
            changeIdx = changeIdx[:-1]
            changePoint_vec[changeIdx] = 1

        elif constantChangePoint == 1:

            changePoints = np.int(np.random.poisson(lambdaV,1))
            nTrials = changePoints*nChangePoints
            changePoint_vec = np.zeros((nTrials))
            changePoint_vec[changePoints::changePoints] = 1
            changeIdx = list(np.cumsum(np.repeat(changePoints, np.sum(changePoint_vec))))
            nTrials = len(changePoint_vec)

    return changePoint_vec, changeIdx, nChangePoints, nTrials


# In[18]:


def genStop(propStopTrials, lambdaStop, nTrials, ssd):

    nStopTrials = np.int(propStopTrials * nTrials)
    stopPoints = np.random.poisson(lambdaStop,nStopTrials) #length
    stopIdx = np.cumsum(stopPoints) #indices

    while stopIdx[-1] >= nTrials:
        stopPoints = np.random.poisson(lambdaStop,nStopTrials)
        stopIdx = np.cumsum(stopPoints)

    #assign delays to the stop indices
    nSamplesPerDelay = np.int(nStopTrials/len(ssd))
    ssd_vec = np.zeros((nTrials))
    ssd_vec[:] = np.nan

    np.random.shuffle(stopIdx)

    #need to reshape, otherwise broadcasting problems
    ssd_repeatedVal = np.reshape(np.repeat(ssd, nSamplesPerDelay),nStopTrials)
    ssd_vec[stopIdx] = ssd_repeatedVal


    return ssd_vec


# In[19]:


def genMuSigma(muMin, muMax, sigmaMin, sigmaMax, changePoint_vec, changeIdx, nTrials):


    nEpochs = np.int(np.sum(changePoint_vec) + 2)

    #more control over separation between mean values of each epoch
    # muDiff = .05 #greater muDiff = less conflict BETWEEN epochs
    # test=np.arange(muMin, muMax, muDiff)
    # muControl_shuffle = np.random.uniform(min(test), max(test), len(test))
    # print(test)

    sigma = np.random.uniform(sigmaMin, sigmaMax, 1)

    if (sigmaMax + muMax) > 1 or (sigmaMax + muMin) < 0:
        raise ValueError("sigmaMax must be lowered to keep range of reward delta >0 and <1")
        return None

    sigma_vec = np.repeat(sigma, nTrials)


    rewardDelta = np.ones((nTrials))

    #this adds 0 and the end idx to change idx
    chgidx = [0]+changeIdx+[nTrials-1]

    #constructs unique interval for each mu
    intervals = [(chgidx[i],chgidx[i+1]) for i in range(len(chgidx)-1)]
    muRewardDelta_vec = np.ones_like(rewardDelta)

    #sample from a normal distribution to generate reward delta for each trial
    while round(np.mean(muRewardDelta_vec),4) != 0:

        mu_p = np.random.uniform(muMin, muMax, np.int(nEpochs/2))
        mu_n = -1*mu_p
        muRewardDelta = np.hstack((mu_p, mu_n))
        np.random.shuffle(muRewardDelta)
        #each iterator is linked to respective range
        #zip iterates over two lists simultaneously
        for interval,mu in zip(intervals,muRewardDelta):
            nElements = interval[1]-interval[0]
            rewardDelta[interval[0]:interval[1]]= np.reshape(np.random.normal(mu,sigma,nElements), (nElements))
            muRewardDelta_vec[interval[0]:interval[1]]= np.reshape(np.repeat(mu,nElements), (nElements))

        #the last element needs to be assigned because of the interval const. above
        #(range is not inclusive)
        rewardDelta[-1] = np.random.normal(muRewardDelta[-1], sigma, 1)
        muRewardDelta_vec[-1] = muRewardDelta[-1]


    return rewardDelta, muRewardDelta_vec, sigma_vec


# In[20]:


def genBaseTargetReward(rewardDelta, nTrials, ssd_vec, sigma_vec, changePoint_vec):

    #reverse engineer the actual values for targets
    rewardDelta_div = 0.5*rewardDelta

    adjustedTargets = np.zeros((nTrials,2))

    #always add half and subtract half of target value
    adjustedTargets[:,0] +=  rewardDelta_div[:]
    adjustedTargets[:,1] -=  rewardDelta_div[:]

    t1_baseReward = adjustedTargets[:,0]
    t2_baseReward = adjustedTargets[:,1]

    return t1_baseReward, t2_baseReward


# In[21]:


def genStaticPlots(ssd_vec, t1_baseReward, t2_baseReward, muRewardDelta_vec,
               rewardDelta):

    filename = input('Enter condition-specific filename for ' +
    'plots and parameter output file (ex: highReliabilityHighAmbigLowConflict): ')

    nTrials = len(ssd_vec)
    x = np.arange(0,nTrials)
    stop_idx = np.isfinite(ssd_vec)


    # plt.figure(figsize=(20, 10))
    # plt.title("raw target reward structure",fontsize = 40)
    # plt.xlabel("trial", fontsize = 30)
    # plt.ylabel("reward", fontsize = 30)
    # plt.scatter(x, t1_baseReward, facecolors='b',edgecolors='b', s = 50)
    # plt.scatter(x, t2_baseReward, facecolors='r',edgecolors='r',s = 50)
    # plt.tick_params(axis='both', which='major', labelsize=20)
    # plt.xlim(0, nTrials)
    # plt.ylim(-1.1, 1.1)
    # plt.legend(["t1", "t2"], fontsize = 20, frameon=True, facecolor = [.7, .7, .7],
    #            loc = 'best')
    #
    # plt.savefig('t1t2_rewardTimecourse_' + filename + '.pdf')


    plt.figure(figsize=(20, 10))
    plt.title("reward structure", fontsize = 40)
    plt.xlabel("trial", fontsize = 30)
    plt.ylabel("$\Delta$ reward [t1 - t2]", fontsize = 30)
    plt.plot(muRewardDelta_vec, 'k')
    plt.scatter(x, rewardDelta,  s = 50,edgecolor='none', facecolor = 'purple')
    plt.plot(x[stop_idx], rewardDelta[stop_idx], 's',color = 'orangered',
            markersize = 15, alpha = .3)
    ax = plt.gca()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlim(0, nTrials)
    plt.ylim(-1.1, 1.1)
    plt.savefig('diff_rewardTimecourse_' + filename + '.pdf')
    plt.show()

    return filename


# In[22]:


def animatePlots(ssd_vec, t1_baseReward, t2_baseReward, muRewardDelta_vec,
               rewardDelta, filename):

    nTrials = len(ssd_vec)
    x = np.arange(0,nTrials)

    # target 1 and 2 reward
    t1t2_reward_fig = plt.figure(figsize=(20, 10))
    plt.plot()
    plt.xlim(0, nTrials)
    plt.ylim(-1.1, 1.1)
    graph, = plt.plot([], [], 'bo')
    graph2, = plt.plot([], [], 'ro')

    plt.title("raw target reward structure",fontsize = 40)
    plt.xlabel("trial", fontsize = 30)
    plt.ylabel("reward", fontsize = 30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(["t1", "t2"], fontsize = 20, frameon=True, facecolor = [.7, .7, .7], loc = 'best')


    def animate(i):
        graph.set_data(x[:i+1], t1_baseReward[:i+1])
        graph2.set_data(x[:i+1], t2_baseReward[:i+1])

        return graph

    anim = FuncAnimation(t1t2_reward_fig, animate, frames=nTrials, interval=10)
#     plt.show()
    HTML(anim.to_html5_video())

    anim.save('t1t2_rewardTimecourse_' + filename +'.mp4',extra_args=['-vcodec', 'libx264'])


    #reward structure diff
    rewardDelta_fig = plt.figure(figsize=(20, 10))
    plt.plot()
    plt.xlim(0, nTrials)
    plt.ylim(-1.1, 1.1)

    graph, = plt.plot([], [], 'k')
    graph2, = plt.plot([], [], 'o', color = 'purple')
    graph3, = plt.plot([], [], 's',color = 'orangered', markersize = 15, alpha = .3)


    plt.title("reward structure", fontsize = 40)
    plt.xlabel("trial", fontsize = 30)
    plt.ylabel("$\Delta$ reward [t1 - t2]", fontsize = 30)


    def animate_data(i):
        graph.set_data(x[:i+1], muRewardDelta_vec[:i+1])
        graph2.set_data(x[:i+1], rewardDelta[:i+1])
        graph3.set_data(x[:i+1], ssd_vec[:i+1])
        return graph

    anim = FuncAnimation(rewardDelta_fig, animate_data, frames=nTrials, interval=10)
#     plt.show()
    HTML(anim.to_html5_video())

    anim.save('diff_rewardTimecourse_' + filename + '.mp4', extra_args=['-vcodec', 'libx264'])

    return None


# In[23]:


#print to file specified by user
def printParameters(t1_baseReward, t2_baseReward, ssd_vec, changePoint_vec, rewardDelta,
                muRewardDelta_vec, sigma_vec, filename):
    taskParameters = np.array((t1_baseReward, t2_baseReward, ssd_vec, changePoint_vec, rewardDelta,
                muRewardDelta_vec, sigma_vec))
    taskParameters = np.matrix.transpose(taskParameters)
    header = ("t1_baseReward, t2_baseReward, ssd, changepoint, rewardDelta, muRewardDelta, sigma")
    np.savetxt(filename + '.csv',taskParameters, header = header, delimiter=',', comments = '', fmt='%f')
    return None

def addRewardProbe(t1_baseReward, t2_baseReward, ssd_vec,
changePoint_vec, rewardDelta, muRewardDelta_vec, sigma_vec):

    highValTargetSelection = int(input('Which target should be high value? [1 or 2?] '))
    nClampTrials = int(input('How many reward-clamped trials? '))

    if (highValTargetSelection != 1 and highValTargetSelection != 2):
        raise ValueError("highValTargetSelection must be either 1 or 2")
        return None
    if nClampTrials <= 0:
        raise ValueError("nClampTrials must be a positive, even integer > 0")
        return None

    rewardClamp = 1
    rewardClamp_vec = np.repeat(rewardClamp, nClampTrials)
    zeroPad = np.zeros((nClampTrials))
    nanPad = np.nan*np.ones((nClampTrials))
    rewardPad = rewardClamp*np.ones((nClampTrials))

    if highValTargetSelection == 1:
        t1_baseReward = np.append(t1_baseReward,rewardClamp_vec)
        t2_baseReward = np.append(t2_baseReward, zeroPad)
    elif highValTargetSelection == 2:
        t2_baseReward = np.append(t2_baseReward,rewardClamp_vec)
        t1_baseReward = np.append(t1_baseReward, zeroPad)

    ssd_vec = np.append(ssd_vec, nanPad)
    muRewardDelta_vec = np.append(muRewardDelta_vec, rewardPad)
    rewardDelta = np.append(rewardDelta, rewardPad)
    sigma_vec = np.append(sigma_vec, zeroPad)
    zeroPad[0] = 1
    changePoint_vec = np.append(changePoint_vec, zeroPad)

    return (t1_baseReward, t2_baseReward, ssd_vec,
    changePoint_vec, rewardDelta, muRewardDelta_vec, sigma_vec)


# In[24]:


def generateTrialStructure(nTrials_approx, muMin, muMax, lambdaV, lambdaStop, ssd, sigmaMin, sigmaMax,
                           propStopTrials, constantChangePoint, animate, rProbe):

    #this is for a constant trial number
    # if nTrials <= 0 or np.mod(nTrials,2) > 0:
    #     raise ValueError("nTrials must be a positive, even integer > 0")
    #     return None
    if sigmaMin < 0:
        raise ValueError("sigmaMin must be a positive integer")
        return None
    if propStopTrials > 1 or propStopTrials < 0:
        raise ValueError("propStopTrials must be a positive value between 0 and 1")
        return None
    if (constantChangePoint != 1 and constantChangePoint != 0):
        raise ValueError("constantChangePoint must be 0 or 1")
        return None

    changePoint_vec, changeIdx, nChangePoints, nTrials = genChangePoint(constantChangePoint, lambdaV, nTrials_approx,
     propStopTrials, ssd)

    ssd_vec = genStop(propStopTrials, lambdaStop, nTrials, ssd)
    rewardDelta, muRewardDelta_vec, sigma_vec = genMuSigma(muMin, muMax, sigmaMin,
                                 sigmaMax, changePoint_vec, changeIdx, nTrials)

    t1_baseReward, t2_baseReward = genBaseTargetReward(rewardDelta, nTrials, ssd_vec,
    sigma_vec, changePoint_vec)

    #small chance that reward diff > 1 or < -1 (potential if sigma is high &
    #sample is drawn from > 1 std). if so, re-sample.
    while np.any(rewardDelta > 1) or np.any(rewardDelta < -1):

        rewardDelta, muRewardDelta_vec, sigma_vec = genMuSigma(muMin, muMax, sigmaMin, sigmaMax,
                         changePoint_vec, changeIdx, nTrials)
        t1_baseReward, t2_baseReward = genBaseTargetReward(rewardDelta, nTrials, ssd_vec,
         sigma_vec, changePoint_vec)

    if rProbe == 1:
        (t1_baseReward, t2_baseReward, ssd_vec,
        changePoint_vec, rewardDelta,
        muRewardDelta_vec, sigma_vec) = addRewardProbe(t1_baseReward, t2_baseReward, ssd_vec,
        changePoint_vec, rewardDelta, muRewardDelta_vec, sigma_vec)

    #print parameters to user-specified filename
    filename = genStaticPlots(ssd_vec, t1_baseReward, t2_baseReward, muRewardDelta_vec,
               rewardDelta)
    printParameters(t1_baseReward, t2_baseReward, ssd_vec, changePoint_vec, rewardDelta,
               muRewardDelta_vec, sigma_vec, filename)
    if animate == 1:
        animatePlots(ssd_vec, t1_baseReward, t2_baseReward, muRewardDelta_vec,
               rewardDelta, filename)


    return t1_baseReward, t2_baseReward, ssd_vec, changePoint_vec, rewardDelta, muRewardDelta_vec, sigma_vec


# In[25]:


#testing


# nTrials = 1000
# lambdaV = 20
# lambdaStop = 10
# ssd = np.array([200, 250, 300, 350, 450])

# propStopTrials = .10
# constantChangePoint = 1
# muMin = .1 #change this to increase conflict within trial
# sigmaMax = .4
# sigmaMin = 0
# animate = 0


# (t1_baseReward, t2_baseReward, ssd_vec, changePoint_vec,
# rewardDelta, muRewardDelta_vec, sigma_vec) = generateTrialStructure(nTrials,
# lambdaV, lambdaStop, ssd, sigmaMin, sigmaMax, propStopTrials,
# constantChangePoint, animate)


# In[ ]:
