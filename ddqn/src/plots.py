# plotting code from Nikhil, Andy

"""
Plotting code for results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd 


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


def policy_radial_heatmap(last_offspring_batch, scal, args):
#def policy_radial_heatmap():
  rad_offset = 10
  theta_width = np.pi/60

  # production code
  n = len(last_offspring_batch)
  m = 50
  z_temp = np.zeros((n,50))
  for i, offspring in enumerate(last_offspring_batch):
    env = SIR_env(cal_model)
    #sample = Sample(cal_model.X_I[-1], cal_model.X_S[-1], objs = [None, None], optgraph_id = -1)
    objs = evaluate_policy(env, offspring.ddqn)
    
    z_temp[i,:] = env.actions


  weights = np.round(np.asarray([sample.weights[0] for sample in scal]), 4)
  a_temp = np.arctan(weights/(1-weights))

  rad = np.linspace(0, m, m+1) + rad_offset

  '''
  # testing code
  n = 6
  m = 50
  rad = np.linspace(0, m, m+1) + rad_offset
  a_temp = np.random.rand(n) * np.pi/2
  z_temp = np.random.uniform(0, 1, (n,m))
  '''

  sort_i = np.argsort(a_temp)
  a_temp = a_temp[sort_i]
  z_temp = z_temp[sort_i,:]

  print(a_temp)
  print(z_temp)

  a = np.array([a_temp[0]])
  z = np.array([z_temp[0,:]])
  for ii in range(1,n):
    a = np.append(a, a_temp[ii-1] + theta_width)
    z = np.append(z, [np.ones(m)*np.nan], axis=0)

    # separates final offsprings with very close weights
    if ((a_temp[ii] - a_temp[ii-1]) < theta_width):
      a_temp[ii] = a_temp[ii-1] + theta_width 

    a = np.append(a, a_temp[ii])
    z = np.append(z, [z_temp[ii,:]], axis=0)

  a = np.append(a, a_temp[-1] + theta_width)
  z = np.append(z, [np.ones(m)*np.nan], axis=0)

  r, th = np.meshgrid(rad, a)

  fig = plt.figure()
  ax = plt.subplot(projection="polar")
  temp = ax.pcolormesh(th, r, z, cmap = 'inferno')

  ax.set_xlim(0, np.pi/2)
  xlabels = ['$f_L$', '$\\frac{\pi}{8}$', '$\\frac{\pi}{4}$', '$\\frac{3\pi}{8}$', '$f_I$']
  xticks = np.linspace(0, np.pi/2, len(xlabels))
  ax.set_xticks(xticks)
  ax.set_xticklabels(xlabels)
  ax.set_ylabel("Day of Episode")
  ylabels = ['']
  [ylabels.append(f"{jj*10}") for jj in range((m//10)+1)]
  yticks = np.linspace(0., m+rad_offset, len(ylabels))
  ax.set_yticks(yticks)
  ax.set_ylim(0,60)
  ax.set_yticklabels(ylabels)
  ax.plot(a, r, ls='none', color = 'k') 
  plt.grid()
  cbar = plt.colorbar(temp, pad=0.15)
  cbar.set_label('Lockdown Severity', rotation=270, labelpad=20)
  plt.savefig('policy_radial_heatmap.pdf')
