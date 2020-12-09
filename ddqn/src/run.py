### Main implementation
from comparison import *
from DDQN import *
from importantClasses import *
from imports import *
from model_cal import *
from mopg import *
from morl import *
from plots import *
from population import *
from run import *
from SIR import *
from util_func import *
from warmup import *

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

    run(args)

def run(args):
    data_arr = torch.load(args.dataset)

    O_I = np.cumsum(data_arr[:,0]) # Cumulative confirmed cases, observation
    ac = data_arr[:,3] # Action time series
    cal_model = model_calibration(O_I, ac)

    beta, gamma = cal_model.model_mls()
    cal_model.model_validate(beta,gamma)

    arguments = {"pgmorl": True,
            "ra": False,
            "pfa": False,
            "moead": False,
            "random": False,
            #"num_seeds": 6, OG
            "num_seeds": 2,
            "num_steps": 2,
            "num_processes": 1,
            "env_name": "sir_env",
            "seed": 0,
            #"num_env_steps": 8000000 ,
            "num_env_steps": 80 ,
            "num_generations": 2 ,
            # "warmup_iter":10, # 100 produces good results ,
            #"warmup_iter":1000, # 100 produces good results , 1000 starts to really give meaningful results
            "warmup_iter":10,
            #"update_iter": 40 ,
            #"update_iter": 1000 ,
            "update_iter": 10 ,
            "min_weight": 0.0 ,
            "max_weight": 1.0 ,
            #"delta_weight": 0.2 ,
            "delta_weight": 0.08, # Daniel: delta should be made small. default was 0.2.
            "eval_num": 10 ,
            "pbuffer_num": 100 ,
            "pbuffer_size": 2 ,
            "selection_method": 'prediction_guided', 
            "num_weights_candidates": 7, 
            #"num_tasks": 6, 
            "num_tasks": 6, # Previously 4
            "sparsity": 1,# default 1.0 ### No influence on changing it to 0.5
            "obj_rms": "",
            "ob_rms": "",
            "obj_num": 2,
            "raw": "",
            "hbolic_init_thresh":"0.01",
            #"hbolic_thresh_factor":"1.1",
            "hbolic_thresh_factor":"2",
            "hbolic_max_thresh":"2e7",
            "save_dir": "./test_policy_save/"
            }


    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    arguments = Struct(**arguments)

    print(arguments.seed)

    morl(arguments) # num_parallel_tasks, warmup_iters, evo_iters, num_generations
    
if __name__ == '__main__':
    main()
