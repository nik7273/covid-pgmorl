"""
Some plotting file used to get plots for the difference in real world data and generated policies.
"""
import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)

import torch
import pandas as pd
import numpy as np
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_parser():
    parser = argparse.ArgumentParser(description='difference')
    parser.add_argument('--obj-weights',
        default="",
        help='Filename containing obj. weights. Ends in .npy')
    parser.add_argument('--save-dir',
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--dataset',
        default=None,
        help='.pt file with lockdown data'
    )
    parser.add_argument(
        '--errors-out',
        default=None,
        help='.csv file to output error between real and DDQN'
    )
    parser.add_argument(
        '--dif-out',
        default=None,
        help='image file to output plot of error between real and DDQN'
    )
    return parser

def main():
    torch.set_default_dtype(torch.float64)

    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    
    # build saving folder
    save_dir = args.save_dir
    try:
        os.makedirs(save_dir, exist_ok = True)
    except OSError:
        pass

    plot_difference(args)

def plot_difference(args):
    weights = np.load(os.path.join(args.save_dir, args.obj_weights))[:,0]
    
    weights = [0.48, 0.796, 0.64, 0.65, 0.505, 0.797]

    print(weights)
    """
    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
      1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1.
      0. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
      1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.
      0. 0.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
      1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
      1. 1.]]
    """
    n= 6
    z_temp = np.zeros((n,50))
    networks = []
    for i in range(6):
        ddqn = torch.load(os.path.join(args.save_dir, f"ddqn{i}.pt"))
        networks.append(ddqn)
        env = SIR_env(cal_model)
        objs = evaluate_policy(env, ddqn)

        z_temp[i,:] = env.actions

    print(z_temp)

    t=90
    D=9
    M = 9986857
    beta_L=0.10484247419161637
    beta_N=0.12361563626079178
    gamma=0.10622938482036372
    data_arr = torch.load(args.dataset)
    data_output = pd.DataFrame(columns = ['org_policy', 'new_policy', 'org', 'res', 'dif', 'weight', 'step'], index = range(50 * 6))

    O_I=np.cumsum(data_arr[:,0])

    X_R=O_I[0:-D]
    X_I=O_I[D:]-O_I[0:-D]
    X_S=M - X_R -X_I


    for i in range(len(networks)):

      weight=weights[i]
      cur_X_R=X_R[t-1]
      cur_X_I=X_I[t-1]  
      cur_X_S=X_S[t-1]
      or_X_R=X_R[t-1]
      or_X_I=X_I[t-1]  
      or_X_S=X_S[t-1]    


      for j in range(50):
        #thepolicy= networks[i].policy(state_normal(np.array([cur_X_I, cur_X_S])))
        thepolicy = z_temp[i, j]
        thepolicy_or=int(data_arr[t+ j,3])
        beta=beta_L
        beta_or=beta_L

        if thepolicy==0:
            beta=beta_N

        if thepolicy_or==0:
            beta_or=beta_N

        e_S = beta*cur_X_S*cur_X_I/M
        e_R = cur_X_I*gamma
        cur_X_S = cur_X_S - e_S
        cur_X_R = cur_X_R + e_R
        cur_X_I = M - cur_X_S - cur_X_R

        e_S = beta_or*or_X_S*or_X_I/M
        e_R = or_X_I*gamma
        or_X_S = or_X_S - e_S
        or_X_R = or_X_R + e_R
        or_X_I = M - or_X_S - or_X_R

        data_output.at[50 * i + j,'res']=cur_X_I+cur_X_R
        data_output.at[50 * i + j,'org_policy']=thepolicy_or
        data_output.at[50 * i + j,'new_policy']=thepolicy
        data_output.at[50 * i + j,'org']=or_X_I+or_X_R
        data_output.at[50 * i + j,'weight']=weight
        data_output.at[50 * i + j,'step']=j
        data_output.at[50 * i + j,'dif']=data_output.at[50* i + j,'res']-data_output.at[50 *i + j,'org']


    data_output.to_csv(args.errors_out, index = False)

    data_output["dif"]=data_output["dif"]/1000
    data = data_output.pivot(index='step', columns='weight', values='dif')
    fig=data.plot(title='Difference between model policy and real policy (Day 90)', figsize = (10, 10))
    fig.set_xlabel("Time steps (days)")
    fig.set_ylabel("Difference Model- Real Infected (thousand)")
    fig = fig.get_figure()
    fig.savefig(args.dif_out)

if __name__ == "__main__":
    main()
