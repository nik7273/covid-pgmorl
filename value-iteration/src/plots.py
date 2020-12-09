"""
Plotting code for results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd 

from class_defs import Sample
from mopg import evaluate_policy, mopg
from sir_model_env import model_calibration, SIR_env

def plot_pareto(last_offspring_batch, model_cal, args):
  """
  Plot Pareto policies for last offspring batch.
  Input(s):
    last_offspring_batch: List[Sample],
    model_cal: model_calibration,
    args
  """
  out_policies = np.zeros((len(last_offspring_batch), 2))
  for i, offspring in enumerate(last_offspring_batch):
    env = SIR_env(model_cal)
    sample = Sample(model_cal.X_I[-1], model_cal.X_S[-1], -1, optgraph_id = -1)
    for j in range(args.mopg_steps):
      new_policy = offspring.policy[j]
      objs = evaluate_policy(new_policy, env, sample, j)
      out_policies[i,0] += objs[0]
      out_policies[i,1] = objs[1]/1000 # / 1000 to get correct num lockdowns
      sample.val_I = np.append(sample.val_I, objs[0])
      sample.val_L = np.append(sample.val_L, objs[1])  

  plt.scatter(out_policies[:, 0], out_policies[:, 1], 20)
  plt.savefig('pareto_last_batch.png')
  plt.show()

def plot_evolution(total_batch, last_offspring_batch, model_cal, args):
  """
  Plot several generations of Pareto policies for comparison.
  Input(s):
    total_batch: List[List[Sample]],
    last_offspring_batch: List[Sample],
    model_cal: model_calibration,
    args
  """
  out_policies = np.zeros((len(total_batch),len(last_offspring_batch), 2))
  for i, offspring_batch in enumerate(total_batch):
    for k, offspring in enumerate(offspring_batch):
      env = SIR_env(model_cal)
      sample = Sample(model_cal.X_I[-1], model_cal.X_S[-1], -1, optgraph_id = -1)
      for j in range(args.mopg_steps):
        new_policy = offspring.policy[j]
        objs = evaluate_policy(new_policy, env, sample, j)
        out_policies[i,k,0] += objs[0]
        out_policies[i,k,1] = objs[1]/1000 #/1000 to get correct num lockdowns
        sample.val_I = np.append(sample.val_I, objs[0])
        sample.val_L = np.append(sample.val_L, objs[1])
  
  plt.scatter(out_policies[0,:, 0], out_policies[0,:, 1], 20)
  # pyplot.scatter(out_policies[2,:, 0], out_policies[2,:, 1], 20)
  # pyplot.scatter(out_policies[5,:, 0], out_policies[5,:, 1], 20)
  plt.savefig('pareto_evolution.png')
  plt.show()

def policy_heatmap(last_offspring_batch,scal,args):
  """
  Plot a heatmap of policy actions (x-axis: offspring, y-axis: time, heat: action)
  Input(s):
    last_offspring_batch: List[Sample]
  """
  
  policies = [sample.policy for sample in last_offspring_batch]
  policies=np.concatenate( policies, axis=0 )
  steps=np.arange(1, args.mopg_steps+1, 1).tolist()* len(last_offspring_batch)
  weights=np.round(np.asarray([sample.weights[0] for sample in scal]),4)

  for i in range(len(weights)):
    theW=weights[i]
    new=0
    for j in range(len(weights)):
      if i!=j and weights[i]==weights[j]:
        new=1
    if new==1:
      weights[i]+=0.0001  


  weights=np.repeat(weights,args.mopg_steps)
  
  data={"steps":steps,"weights":weights,"policies":policies}
  data=pd.DataFrame(data)
  data.to_csv(r'policies.csv', index = False)
  data=data.pivot("weights", "steps", "policies")

  ax = sns.heatmap(data)
  ax.invert_yaxis()
  ax.hlines(
    list(range(len(last_offspring_batch))), 
    *ax.get_xlim(), 
    color='white',
  )
  ax.set(xlabel='Time step (days)', ylabel='Output Policy (weight for infected objective)')
  plt.savefig('policy_heatmap.png')
  plt.show()




