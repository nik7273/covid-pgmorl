# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:55:46 2020

@author: dfotero
"""

import numpy as np
from copy import copy, deepcopy
from scipy.stats import poisson
from scipy.stats import binom

env=SIR_env(calibration)

#----------MOPG-------------------
# See Alg. 2 in Xu et al.
#INPUT:
#-env: enviroment
#-tasks:group of tasks where each task has:
#       - X_I: current infected
#       - X_S: current Susceptible
#       - pol: Current policy (1: lockdown, 0:no lockdown)  
#       - w_I: Current weight for the objective to minimize infected 
#       - w_L: Current weight value for the objective to minimize lockdowns
#       - val_I: Current optimal value for the objective to minimize infected 
#       - val_L: Current optimal value for the objective to minimize lockdowns
#-m: number of iterations for mopg
#OUTPUT:
#-offspring population P'
def mopg(env, tasks, m):
    offspring = []
    for i in range(len(tasks)):
        curr_task = copy(tasks[i])
        new_policy = policy_gradient(env, curr_task, m)
        objs = evaluate_policy(
            new_policy,
            env,
            curr_task.sample
        )
        # theTask.X_I,
        # theTask.X_S,
        # theTask.val_I[theTask.X_I,theTask.X_S],
        # theTask.val_L[theTask.X_I,theTask.X_S]
        curr_task.sample.val_I[theTask.X_I,theTask.X_S] = objs[0]
        curr_task.sample.val_L[theTask.X_I,theTask.X_S] = objs[1]
        curr_task.pol[theTask.X_I,theTask.X_S] = new_policy
        offspring.append(curr_task.sample)
    return offspring

#----------evaluate_policy-------------------
# Evaluate a policy and returns the values
#INPUT:
#-newPol:policy to evaluate
#-env: enviroment
#-X_I: current infected
#-X_S: current Susceptible
#-currentV_I: Current value for the objective to minimize infected 
#-currentV_L: Current value for the objective to minimize lockdowns 
#OUTPUT:
#-val_I:New value for the objective to minimize infecte
#-val_L:New value for the objective to minimize lockdowns 
def evaluate_policy(new_policy, env, sample):#X_I,X_S,currentV_I,currentV_L):
    env.time_step(new_policy)
    
    X_I, X_S = sample.X_I, sample.X_S
    currentV_I = sample.val_I
    currentV_L = sample.val_L
    
    meanX_S, meanX_I, meanX_R = env.sample_stochastic()
    errX_S, errX_I, errX_R = env.get_error()
    
    val_I=meanX_I
    val_L=new_policy
    
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

    objs = [val_I, val_L]
    return objs

#----------policy_gradient-------------------
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
#-thePol:policy, 0 or 1

def policy_gradient(env,sample,m):
    thePol=task.pol
    X_I=sample.X_I
    X_S=sample.X_S
    
    val_I=sample.val_I[X_I,X_S]
    val_L=sample.val_L[X_I,X_S]
    
    for i in range(1,m):
        #1 stands for lockdown, 0 no lockdown
        valL_I,valL_L=evaluate_policy(1,env,X_I,X_S,val_I,val_L)
        valL=task.w_I*valL_I+task.w_L*valL_L
        
        valN_I,valN_L=evaluate_policy(0,env,X_I,X_S,val_I,val_L)
        
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
