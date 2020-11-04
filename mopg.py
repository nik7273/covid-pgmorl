# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:55:46 2020

@author: dfotero
"""

import numpy as np
from scipy.stats import poisson
from scipy.stats import binom

env=SIR_env(calibration)

def mopg(tasks,m,R):
    for i in range(len(tasks)):
        theTask=tasks[i]
        F= evalPol(theTask.pol)
        newPol = polGrad(theTask,m)
        newF= evalPol(newPol)
        theTask.pol=newPol
        tasks[i]=theTask
    return tasks




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
            
            val_I+=probI*probR*currentV_I[X_I+i-j-1,X_S-i-1]
            val_L+=probI*probR*currentV_L[X_I+i-j-1,X_S-i-1]
    
    return val_I,val_L


def polGrad(task,m):
    
