"""
Multi-Objective Policy Optimization via DDQN.
Outputs:
a population P_new
"""
def mopg(env, task_list, m, eval_num):
  '''
  Inputs:
    env: the environment object;
    task_list: the current list tasks to solve
    m: number of policy update iterations
    eval_num: number of evaluations of policy_evaluation
  '''
  P_offspring = []#initialize_population()
  print("MOPG Objective values")
  for i, taskobj in enumerate(task_list):
    
    curr_task = deepcopy(taskobj)

    old_ddqn = taskobj.sample.ddqn
    weights = taskobj.scalarization
    
    
    # gradient update for m steps
    new_ddqn = deepcopy(old_ddqn)
    new_ddqn = gradient_update(env, new_ddqn, weights, m) 

    # objs = evaluate_policy(env, new_ddqn, num_evals = 50)
    objs = evaluate_policy(env, new_ddqn, eval_num)
    print('task id = {}, objs = {}, weighted objs = {}'.format(i, objs, np.matmul(objs,weights.weights.numpy())))

    '''
    Data organization still need to be confirmed here
    '''
    curr_task.sample.val_I = objs[0]
    curr_task.sample.val_L = objs[1]
    curr_task.sample.X_I = env.X_I[-1]
    curr_task.sample.X_S = env.X_S[-1]
    curr_task.sample.objs = objs
    curr_task.sample.ddqn = new_ddqn
    curr_task.sample.policy = env.actions[-1]
    P_offspring.append(curr_task.sample)
    
  return P_offspring

def gradient_update(sir_env, ddqn, weights, m):
    """
    Inputs:
      sir_env: SIR environment model
      ddqn: a DDQN policy network
      weights: weights of objective functions
      m: number of policy iterations
    Outputs:
      ddqn: a network with updated parameters
    """
    rewards = 0    
    for i in range(m):
        state, done = sir_env.reset()
        episodic_reward = np.zeros(2)

        state[:-1] = state_normal(state[:-1])
        while not done:

            action = ddqn.select_action(state) 

            (X_S, X_I, X_R), reward, done, info = sir_env.timeStep(action)

            next_state_ = state_normal(np.array([X_S, X_I]))
            next_state = np.array([next_state_[0], next_state_[1], X_R])

            episodic_reward += reward
            sign = 1 if done else 0
            reward_n = reward_normal(reward)
            ddqn.train(state, action, reward_n, next_state, sign, weights.weights.numpy())
            state = next_state

    return ddqn

def state_normal(state):
    '''
    To normalize the input to neural network to be in [0, 1]
    '''
    ### X_S in Population, X_I in [517 1100] - [500, 5000]
    # Total population M = 9986857
    state_min = np.array([9986857 - 5000, 500])
    state_scale = np.array([4500, 500])
    state_n = (state - state_min) / state_scale
    return state_n

def reward_normal(reward):
    '''
    To normalize the reward to be in [0,1]
    '''
    lower_bound = np.array([-100, -1])
    scale = np.array([50, 1])

    reward_n = (reward - lower_bound)/scale # Make the reward_n to be in [0,1]^n space
    return reward_n

def evaluate_policy(sir_env, ddqn, num_evals = 50):
    """
    Inputs:
      ddqn: a ddqn policy network

    Outputs:
      Each reward
    """
    eval_scorelist=[]
    for i in range(num_evals):
        state, done = sir_env.reset()
        state = np.array(state[0:2]).astype(float)
       
        episodic_reward = np.zeros(2)
        while not done:
            action = ddqn.policy(state_normal(state))

            (X_S, X_I, X_R), reward, done, info = sir_env.timeStep(action)

            next_state = np.array([X_S, X_I])

            episodic_reward += reward_normal(reward)
            sign = 1 if done else 0
            state = deepcopy(next_state).astype(float)
        # evaluate how good this action is:
        eval_score = episodic_reward
        eval_scorelist.append(eval_score)
        # print(f'yeo {i}')

    return np.sum(np.array(eval_scorelist),0) / num_evals 

def compute_advantages(rewards, baselines):
    max_episode_length = rewards.shape[-1]
    gamma = 0.99
    gae_lambda = 0.3
    adv_filter = torch.full((1, 1, 1, max_episode_length - 1),
                            gamma * gae_lambda,
                            dtype=torch.float)
    adv_filter = torch.cumprod(F.pad(adv_filter, (1, 0), value=1), dim=-1)
    deltas = (rewards + gamma * F.pad(baselines, (0, 1))[:, 1:] - baselines)
    deltas = F.pad(deltas, (0, max_episode_length - 1)).unsqueeze(0).unsqueeze(0)
    advantages = F.conv2d(deltas, adv_filter, stride=1).reshape(rewards.shape)
    return advantages
