import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)

from copy import deepcopy
from class_defs import EP, Sample, WeightedSumScalarization, OptGraph, Task
from mopo import evaluate_policy, mopo
from population import Population
from utils import update_ep, generate_weights_batch_dfs, print_info
from sir_model_env import model_calibration, SIR_env
from output import *
from plots import *

import numpy as np
import torch
import torch.optim as optim

import time

def initialize_warmup_batch(args, model_cal, device):
    """
    Training policies during warmup stage
    """
    # using evenly distributed weights for warm-up stage
    weights_batch = []
    generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
    sample_batch = []
    scalarization_batch = []

    temp_env = SIR_env(model_cal) # temp_env is only used for initialization

    for weights in weights_batch:
        
        scalarization = WeightedSumScalarization(num_objs = args.obj_num, weights = weights)
        sample = Sample(model_cal.X_I[-1], model_cal.X_S[-1], -1, optgraph_id = -1)
        objs = evaluate_policy(sample.policy, temp_env, sample, 0)
        sample.objs = objs
        print("XI", sample.X_I)

        sample_batch.append(sample)
        scalarization_batch.append(scalarization)

    return sample_batch, scalarization_batch


def run(args):
    print("In run")
    """
    Runs the entire MORL algorithm. (See Alg. 1 in Xu et al.)
    """
    # Torch stuff
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # Load lockdown dataset
    data_arr = torch.load(args.dataset)
    O_I = np.cumsum(data_arr[:,0])
    print("O_I",O_I)
    AC = data_arr[:,3] # Action time series
    model_cal = model_calibration(O_I, AC)
    model_cal.model_mls()

    # Initialization
    scalarization_template = WeightedSumScalarization(num_objs = args.obj_num, weights = np.ones(args.obj_num) / args.obj_num)
    total_num_updates = args.total_num_updates
    # start_time = time.time()
    
    external_pareto = EP()
    population = Population()
    opt_graph = OptGraph()
    
    selected_tasks, scalarization_batch = initialize_warmup_batch(args, model_cal, device)
    rl_num_updates = args.warmup_iter
    for sample, scalarization in zip(selected_tasks, scalarization_batch):
        sample.optgraph_id = opt_graph.insert(deepcopy(scalarization.weights), deepcopy(sample.objs), -1)

    episode = 0
    iteration = 0
    print("Done initializing")
    total_batch=[]
    while iteration < total_num_updates:
        print(f"In iteration {iteration}")
        if episode == 0:
            print_info('\n------------------------------- Warm-up Stage -------------------------------')    
        else:
            print_info('\n-------------------- Evolutionary Stage: Generation {:3} --------------------'.format(episode))

        episode += 1
        
        offspring_batch = np.array([])

        # --------------------> RL Optimization <-------------------- #
        # compose task for each elite
        task_batch = []
        
        for selected, scalarization in \
                zip(selected_tasks, scalarization_batch):
            task_batch.append(Task(selected, scalarization)) # each task is a (policy, weight)
            task_batch[-1].sample.index_task=len(task_batch)-1

        all_offspring_batch = []
        
        for task_id, task in enumerate(task_batch):
            env = SIR_env(model_cal)
            offsprings_task = mopo(env, task, rl_num_updates, args)
            all_offspring_batch.append(offsprings_task)
        
        # put all intermediate policies into all_sample_batch for EP update
        all_sample_batch = [] 
        last_offspring_batch = [None] * len(task_batch) 
        # only the policies with iteration % update_iter = 0 are inserted into offspring_batch for population update
        # after warm-up stage, it's equivalent to the last_offspring_batch
        # opt_graph.weights[all_offspring_batch[2][-1].optgraph_id]
        offspring_batch = [] 
        for task_id in range(len(task_batch)):
            offsprings = all_offspring_batch[task_id]
            prev_node_id = task_batch[task_id].sample.optgraph_id
            opt_weights = deepcopy(task_batch[task_id].scalarization.weights).detach().numpy()
            for i, sample in enumerate(offsprings):
                all_sample_batch.append(sample)
                if (i + 1) % args.update_iter == 0:
                    prev_node_id = opt_graph.insert(opt_weights, deepcopy(sample.objs), prev_node_id)
                    sample.optgraph_id = prev_node_id
                    offspring_batch.append(sample)
            last_offspring_batch[task_id] = offsprings[-1]

        # finished_event.set()
        print_info('Batch Results:')
        for i in range(len(last_offspring_batch)):
            print_info(
              'objs = {}, weight = {}'.format(
                last_offspring_batch[i].objs,
                task_batch[last_offspring_batch[i].index_task].scalarization.weights
              )
            )
        
        total_batch.append(last_offspring_batch)
        # ----------------------> Update EP <------------------------ #
        external_pareto.update(all_sample_batch)
        population.population.extend(offspring_batch)
        # -------------------> Task Selection for Next Stage/Evaluation <--------------------- #
        last_scalarization_batch=scalarization_batch
        selected_samples, scalarization_batch, predicted_offspring_objs = \
            population.prediction_guided_task_selection(args, external_pareto, opt_graph, scalarization_template)
        print('Selected Tasks:')
        for i in range(len(selected_samples)):
            print_info('objs = {}, weight = {}'.format(selected_samples[i].objs, scalarization_batch[i].weights))
        
        iteration = min(iteration + rl_num_updates, total_num_updates)

        rl_num_updates = args.update_iter

    print('begin evaluation and plots')

    policy_heatmap(last_offspring_batch,last_scalarization_batch,args)
    plot_pareto(last_offspring_batch, model_cal, args)
    
    print("DONE!")

