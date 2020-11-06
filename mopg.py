# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:55:46 2020

@author: dfotero, devrajn
"""

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
#-val_I:New value for the objective to minimize infecte
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
    scalarization = task.scalarization
    actor_critic, agent = task.sample.actor_critic, task.sample.agent

    weights_str = (args.obj_num * '_{:.3f}').format(*task.scalarization.weights)

    # TODO: Need to figure out how to emulate vec_envs (line 67 in original mopg.py)

    # The following is only useful if we can get our environment to support an application of PPO
    # build rollouts data structure for observations
    rollouts = RolloutStorage(num_steps = args.num_steps, num_processes = args.num_processes,
                              obs_shape = envs.observation_space.shape, action_space = envs.action_space,
                              recurrent_hidden_state_size = actor_critic.recurrent_hidden_state_size, obj_num=args.obj_num)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode_lens = deque(maxlen=10)
    episode_objs = deque(maxlen=10)   # for each cost component we care
    episode_obj = np.array([None] * args.num_processes)

    total_num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    start_iter, final_iter = iteration, min(iteration + num_updates, total_num_updates)
    for j in range(start_iter, final_iter):
        torch.manual_seed(j)
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule( \
                agent.optimizer, j * args.lr_decay_ratio, \
                total_num_updates, args.lr)
        
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            obs, _, done, infos = envs.step(action)
            obj_tensor = torch.zeros([args.num_processes, args.obj_num])

            for idx, info in enumerate(infos):
                obj_tensor[idx] = torch.from_numpy(info['obj'])
                episode_obj[idx] = info['obj_raw'] if episode_obj[idx] is None else episode_obj[idx] + info['obj_raw']
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_lens.append(info['episode']['l'])
                    if episode_obj[idx] is not None:
                        episode_objs.append(episode_obj[idx])
                        episode_obj[idx] = None

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, obj_tensor, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        obj_rms_var = envs.obj_rms.var if envs.obj_rms is not None else None

        value_loss, action_loss, dist_entropy = agent.update(rollouts, scalarization, obj_rms_var)

        rollouts.after_update()

        # evaluate new sample
        sample = Sample(env_params, deepcopy(actor_critic), deepcopy(agent))
        objs = evaluation(args, sample)
        sample.objs = objs
        offspring_batch.append(sample)

        if args.rl_log_interval > 0 and (j + 1) % args.rl_log_interval == 0 and len(episode_rewards) > 1:
            if task_id == 0:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                end = time.time()
                print(
                    "[RL] Updates {}, num timesteps {}, FPS {}, time {:.2f} seconds"
                    .format(j + 1, total_num_steps,
                            int(total_num_steps / (end - start_time)),
                            end - start_time))

        # put results back every update_iter iterations, to avoid the multi-processing crash
        if (j + 1) % args.update_iter == 0 or j == final_iter - 1:
            offspring_batch = np.array(offspring_batch)
            results = {}
            results['task_id'] = task_id
            results['offspring_batch'] = offspring_batch
            if j == final_iter - 1:
                results['done'] = True
            else:
                results['done'] = False
            results_queue.put(results)
            offspring_batch = []

    done_event.wait()
