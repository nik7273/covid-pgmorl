# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:55:46 2020

@author: dfotero, devrajn
"""

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from vectorized_envs import make_env, make_vec_envs # differs from a2c_ppo_acktr

from sample import Sample

import numpy as np
from scipy.stats import poisson
from scipy.stats import binom

env=SIR_env(calibration)

#----------MOPG-------------------
#INPUT:
#-env: enviroment
#-tasks:group of tasks where each task has:
#       - X_I: current infected
#       - X_S: current Susceptible
#       - pol: Current policy (1: lockdown, 0:no lockdown)  
#       - w_I: Current weight for the objective to minimize infected 
#       - w_L: Current weight value for the objective to minimize lockdowns
# -currentF: has the information of the values for each objective
#       - val_I: Current optimal value for the objective to minimize infected 
#       - val_L: Current optimal value for the objective to minimize lockdowns
#-m: number of iterations for mopg
#OUTPUT:
#-update tasks (policy)
#-update F(values)
def mopg(env,tasks,m,currentF):
    for i in range(len(tasks)):
        theTask=tasks[i]
        newPol = polGrad(env,theTask,currentF,m)
        currentF.val_I[theTask.X_I,theTask.X_S], currentF.val_L[theTask.X_I,theTask.X_S] =\
          evalPol(newPol,env,theTask.X_I,theTask.X_S,currentF.val_I[theTask.X_I,theTask.X_S],currentF.val_L[theTask.X_I,theTask.X_S])
        theTask.pol=newPol
        tasks[i]=theTask
    return tasks,currentF

#----------evalPol-------------------
# Evaluate a policy and returns the values
#INPUT:
#-newPol:policy to evaluate
#-env: enviroment
#-X_I: current infected
#-X_S: current Susceptible
#-currentV_I: Current value for the objective to minimize infected 
#-currentV_L: Current value for the objective to minimize lockdowns 
#OUTPUT:
#-val_I:New value for the objective to minimize infected
#-val_L:New value for the objective to minimize lockdowns 
def evalPol(newPol,env,X_I,X_S,currentV_I,currenV_L):
    env.time_step(newPol)
    meanX_S, meanX_I, meanX_R = env.sample_stochastic()
    errX_S, errX_I, errX_R = env.get_error()
    
    val_I=meanX_I
    val_L=newPol
    
    lowXS=max(round(meanX_S-errX_S,0),0)
    uppXS=min(round(meanX_S+errX_S,0),env.M)
    
    lowXI=max(round(meanX_I-errX_I,0),0)
    uppXI=min(round(meanX_I+errX_I,0),env.M)
    
    lowXR=max(round(meanX_R-errX_R,0),0)
    uppXR=min(round(meanX_R+errX_R,0),env.M)
    
    lowI=max(X_S-uppXS,0)
    uppI=min(X_S,X_S-lowXS)
    
    lowR=max(lowXR-(env.M-X_I-X_S),0)
    uppR=min(X_I,uppXR-(env.M-X_I-X_S))
    
    for i in range(lowI,uppI):
        for j in range(lowR,uppR):
            
            probI=poisson.pmf(i,env.beta)
            probR=binom.pmf(j,uppR,env.gamma)
            
            if i==lowI:
                probI=poisson.cdf(i,env.beta)
                
            if i==uppI:
                probI=1-poisson.cdf(i-1,env.beta)
            
            if j==lowR:
                probR=binom.cdf(j,uppR,env.gamma)
            if j==uppR:
                probR=1-binom.cdf(j-1,uppR,env.gamma)
            
            val_I+=0.97*probI*probR*currentV_I[X_I+i-j-1,X_S-i-1]
            val_L+=0.97*probI*probR*currentV_L[X_I+i-j-1,X_S-i-1]
    
    return val_I,val_L

#----------polGrad-------------------
# Returns new policy
#INPUT:
#-env: enviroment
#-tasks:group of tasks where each task has:
#       - X_I: current infected
#       - X_S: current Susceptible
#       - pol: Current policy (1: lockdown, 0:no lockdown)  
#       - w_I: Current weight for the objective to minimize infected 
#       - w_L: Current weight value for the objective to minimize lockdowns
# -currentF: has the information of the values for each objective
#       - val_I: Current optimal value for the objective to minimize infected 
#       - val_L: Current optimal value for the objective to minimize lockdowns
#-m: number of iterations for mopg
#OUTPUT:
#-thePol:policy

def polGrad(env,task,currentF,m):
    thePol=task.pol
    X_I=task.X_I
    X_S=task.X_S
    
    val_I=currentF.val_I[X_I,X_S]
    val_L=currentF.val_L[X_I,X_S]
    
    for i in range(1,m):
        #1 stands for lockdown, 0 no lockdown
        valL_I,valL_L=evalPol(1,env,X_I,X_S,val_I,val_L)
        valL=task.w_I*valL_I+task.w_L*valL_L
        
        valN_I,valN_L=evalPol(0,env,X_I,X_S,val_I,val_L)
        
        valN=task.w_I*valN_I+task.w_L*valN_L
        
        if valL<=valN:
            thePol=1
            val_I=valL_I
            val_L=valL_L
        else:
            thePol=0
            val_I=valN_I
            val_L=valN_L
    
    return thePol

#------------mopg_worker------------------#
# Runs MOPG for a candidate task
# INPUT:
#-args: top-level arguments
#-task_id: unique task identification
#-task: (policy, weight) pair
#-device: device for torch
#-iteration: iteration of the evolutionary algorithm (Alg. 1)
#-num_updates: number of optim steps
#-start_time: time started worker
#-results_queue: offspring for worker (for multiprocessing)
#-done_event: signal waiting to multiprocess allocator

def mopg_worker(args, task_id, task, device, iteration, num_updates, start_time, results_queue, done_event):

    # actor_critic = task.sample.actor_critic
    # agent = task.sample.agent
    pass
    
    
    
    # TODO: reimplement
    
