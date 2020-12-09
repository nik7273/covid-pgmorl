"""
Multi-Objective Policy Optimization thru Value Iteration.
"""

import numpy as np
from copy import copy, deepcopy
from scipy.stats import poisson
from scipy.stats import binom
from class_defs import Sample

def mopo(env, task, m, args):
  """
  Multi-objective Policy Optimization
  Adaptation of MOPG (Alg. 2 in Xu et al.)
  Input(s):
    env: SIR_env,
    task: Task,
    m: int representing number of steps to look forward,
    args
  Output(s):
    offspring: List[Task] of offspring tasks
  """
  offspring = []
  for i in range(m):
    curr_task = task
    curr_policy = curr_task.sample.policy
    curr_task.sample.policy = []
    env_task = deepcopy(env)
    for j in range(args.mopg_steps):
      new_policy = policy_optimization(env_task, curr_task, curr_policy, j, args.mopg_steps)
      objs = evaluate_policy(
        new_policy,
        env_task,
        curr_task.sample,
        j
      )
      curr_task.sample.val_I = np.append(curr_task.sample.val_I, objs[0])
      curr_task.sample.val_L = np.append(curr_task.sample.val_L, objs[1])
      curr_task.sample.objs = objs
      curr_task.sample.policy.append(new_policy)
      print(
        "weight",
        curr_task.scalarization.weights,
        "step:",j,
        "valI:",objs[0],
        "valL:",objs[1],
        "policy:",new_policy
      )
    offspring.append(curr_task.sample)
  return offspring

def evaluate_policy(new_policy, env, sample, start):
  """
  Evaluates an input policy in a temporary environment.
  Input(s):
    new_policy: int representing whether to lock down,
    env: SIR_env,
    sample: Sample,
    start: int (is 0 when we're on the first iteration of MORL)
  Output(s):
    objs: ndarray([perf. for objective I, perf. for objective L])
  """
  env.timeStep(new_policy)
  X_I, X_S = env.X_I[-1], env.X_S[-1]

  currentV_I = sample.val_I
  currentV_L = sample.val_L

  meanX_S, meanX_I, meanX_R = env.sampleStochastic()
  errX_S, errX_I, errX_R = env.getError()

  val_I = meanX_I + meanX_R

  if(start == 0):
    val_L = 0
  else:
    val_L = sample.val_L[-1]

  lowXS = max(round(meanX_S - errX_S, 0), 0)
  uppXS = min(round(meanX_S + errX_S, 0), env.M)
  
  lowXR = max(round(meanX_R - errX_R, 0), 0)
  uppXR = min(round(meanX_R + errX_R, 0), env.M)
  
  lowI = int(max(X_S - uppXS, 0))
  uppI = int(min(X_S, X_S - lowXS))
  
  lowR = int(max(lowXR - (env.M - X_I - X_S), 0))
  uppR = int(min(X_I, uppXR - (env.M - X_I - X_S)))
  
  for i in range(lowI,uppI):
          
    probI=poisson.pmf(i,env.beta[new_policy])
          
    if i==lowI:
        probI = poisson.cdf(i, env.beta[new_policy])
          
    if i==uppI:
        probI = 1 - poisson.cdf(i-1, env.beta[new_policy])

    val_I+=probI*i
    
  val_L+=new_policy*1000 # scaling factor to even out with val_I
  
  print("valI:",val_I)
  objs = np.asarray([val_I, val_L])
  return objs

def policy_optimization(env, task, curr_policy, j, mopg_steps):
    """
    Runs policy optimization on task to obtain best policy.
    Note: In this implementation, it's value iteration.
    Input(s):
      env: SIR_env,
      task: Task,
      curr_policy: List[int] (0 or 1),
      j: int (index of current MOPO iteration),
      mopg_steps: int (num times to run policy gradient)
    Output(s):
      best_policy: best policy to use based on results of policy optim.
    """
    
    X_I = task.sample.X_I
    X_S = task.sample.X_S
    val_obj_I = task.sample.val_I[-1] #minimizing infections
    val_obj_L = task.sample.val_L[-1] #minimizing lockdowns
    best_policy = -1

    sim_valN = 0
    sim_valL = 0
     
    env_1 = deepcopy(env)     
    env_0 = deepcopy(env)

    X_I_N, X_I_L = X_I, X_I
    X_S_N, X_S_L = X_S, X_S

    X_I_L, X_S_L, sim_valL = pg_helper(X_I_L, X_S_L, sim_valL, task, val_obj_I, val_obj_L, 1, env_1, 0)
    X_I_N, X_S_N, sim_valN = pg_helper(X_I_N, X_S_N, sim_valN, task, val_obj_I, val_obj_L, 0, env_0, 0)

    for i in range(1,mopg_steps):
      #1 stands for lockdown, 0 no lockdown
      if curr_policy == 0:
        X_I_L, X_S_L, sim_valL = pg_helper(X_I_L, X_S_L, sim_valL, task, val_obj_I, val_obj_L, 1, env_1, i)
        X_I_N, X_S_N, sim_valN = pg_helper(X_I_N, X_S_N, sim_valN, task, val_obj_I, val_obj_L, 0, env_0, i)
      else:
        X_I_L, X_S_L, sim_valL =\
          pg_helper(X_I_L, X_S_L, sim_valL, task, val_obj_I, val_obj_L, curr_policy[min(i+j-1, mopg_steps-1)], env_1, i)
        X_I_N, X_S_N, sim_valN =\
          pg_helper(X_I_N, X_S_N, sim_valN, task, val_obj_I, val_obj_L, curr_policy[min(i+j-1, mopg_steps-1)], env_0, i)
          
    if sim_valL <= sim_valN:
        best_policy = 1
    else:
        best_policy = 0
    env.timeStep(best_policy)   
    return best_policy
  
def pg_helper(X_I, X_S, sim_val, task, val_obj_I, val_obj_L, test_policy, env, idx):
    """
    Evaluates sample and yields state.
    Input(s):
      X_I: int (num. infected),
      X_S: int (num. susceptible),
      sim_val: int (simulated value),
      task: Task,
      val_obj_I: int (task val I),
      val_obj_L: int (task val L),
      test_policy: int (1 or 0, depending on lockdown decision)
      env: SIR_env,
      idx: int
    Output(s):
      updated X_I, X_S, sim_val based on test_policy evaluation
    """
    temp_sample = Sample(X_I, X_S, task.sample.index_task, np.asarray([val_obj_I, val_obj_L]))
    objs = evaluate_policy(test_policy, env, temp_sample, idx)
    sim_val += task.scalarization.evaluate(objs)
    X_I, X_S = env.X_I[-1], env.X_S[-1]
    return X_I, X_S, sim_val

