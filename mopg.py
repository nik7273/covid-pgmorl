import numpy as np
from copy import copy, deepcopy
from scipy.stats import poisson
from scipy.stats import binom
from class_defs import Sample

#----------MOPG-------------------
# See Alg. 2 in Xu et al.
#INPUT:
#-env: enviroment
#-tasks:group of Tasks where each task's sample has:
#       - X_I: current infected
#       - X_S: current Susceptible
#       - val_I: Current optimal value for the objective to minimize infected 
#       - val_L: Current optimal value for the objective to minimize lockdowns
#       - policy: Current policy (1: lockdown, 0:no lockdown)  
# and each Task additionally includes a scalarization_batch with:
#       - w_I: Current weight for the objective to minimize infected 
#       - w_L: Current weight value for the objective to minimize lockdowns

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
        curr_task.sample.val_I[curr_task.sample.X_I,curr_task.sample.X_S] = objs[0]
        curr_task.sample.val_L[curr_task.sample.X_I,curr_task.sample.X_S] = objs[1]
        curr_task.pol[curr_task.sample.X_I,curr_task.sample.X_S] = new_policy
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
def evaluate_policy(new_policy, env, sample):
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
    
    # lowXI=max(round(meanX_I-errX_I,0),0)
    # uppXI=min(round(meanX_I+errX_I,0),env.M)
    
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

def policy_gradient(env, task, m):
    
    X_I = task.sample.X_I
    X_S = task.sample.X_S
    val_I = task.sample.val_I[X_I,X_S]
    val_L = task.sample.val_L[X_I,X_S]
    current_policy = -1
    for _ in range(1,m):
        #1 stands for lockdown, 0 no lockdown   
        env_1 = deepcopy(env)     
        temp_sample_L = Sample(X_I, X_S, [val_I, val_L])
        objs_L = evaluate_policy(1, env_1, temp_sample_L)
        valL = task.scalarization.evaluate(objs_L)

        env_0 = deepcopy(env)
        temp_sample_N = Sample(X_I, X_S, [val_I, val_L])
        objs_N = evaluate_policy(0, env_0, temp_sample_N)
        valN = task.scalarization.evaluate(objs_N)
        
        valL_I, valL_L = objs_L[0], objs_L[1]
        valN_I, valN_L = objs_N[0], objs_N[1]
        if valL <= valN:
            current_policy=1
            env = env_1
            val_I = valL_I
            val_L = valL_L
        else:
            current_policy=0
            env = env_0
            val_I = valN_I
            val_L = valN_L
    
    return current_policy
