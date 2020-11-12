import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

from copy import deepcopy
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.model import Policy

from utils import generate_weights_batch_dfs
from scalarization_methods import WeightedSumScalarization
from mopg import evaluation, mopg_worker
from population import Population
from opt_graph import OptGraph

import torch.optim as optim
from multiprocessing import Process, Queue, Event


def generate_weights_batch_dfs(i, obj_num, min_weight, max_weight, delta_weight, weight, weights_batch):
    if i == obj_num - 1:
        weight.append(1.0 - np.sum(weight[0:i]))
        weights_batch.append(deepcopy(weight))
        weight = weight[0:i]
        return
    w = min_weight
    while w < max_weight + 0.5 * delta_weight and np.sum(weight[0:i]) + w < 1.0 + 0.5 * delta_weight:
        weight.append(w)
        generate_weights_batch_dfs(i + 1, obj_num, min_weight, max_weight, delta_weight, weight, weights_batch)
        weight = weight[0:i]
        w += delta_weight


'''
Each Sample is a policy which contains the actor_critic, agent status and running mean std info.
The algorithm can pick any sample to resume its training process or train with another optimization direction
through those information.
Each Sample is indexed by a unique optgraph_id
'''
class Sample:
    def __init__(self, actor_critic, agent, objs = None, optgraph_id = None):
        self.actor_critic = actor_critic
        self.agent = agent
        self.link_policy_agent()
        self.objs = objs
        self.optgraph_id = optgraph_id

    @classmethod
    def copy_from(cls, sample):
        actor_critic = deepcopy(sample.actor_critic)
        agent = deepcopy(sample.agent)
        objs = deepcopy(sample.objs)
        optgraph_id = sample.optgraph_id
        return cls(actor_critic, agent, objs, optgraph_id)

    def link_policy_agent(self):
        self.agent.actor_critic = self.actor_critic
        optim_state_dict = deepcopy(self.agent.optimizer.state_dict())
        self.agent.optimizer = optim.Adam(self.actor_critic.parameters(), lr = 3e-4, eps = 1e-5)
        self.agent.optimizer.load_state_dict(optim_state_dict)


# TODO: fix this
def initialize_warmup_batch(args, device):
    """
    Training policies during warmup stage
    """
    
    # using evenly distributed weights for warm-up stage
    weights_batch = []
    generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
    sample_batch = []
    scalarization_batch = []

    temp_env = SIR_env(args.model_cal) # temp_env is only used for initialization

    for weights in weights_batch:
        actor_critic = Policy(
            temp_env.observation_space.shape,
            temp_env.action_space,
            base_kwargs={'layernorm' : args.layernorm},
            obj_num=args.obj_num)

        actor_critic.to(device).double()

        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=1e-5,
            max_grad_norm=args.max_grad_norm)

        scalarization = WeightedSumScalarization(num_objs = args.obj_num, weights = weights)

        sample = Sample(actor_critic, agent, optgraph_id = -1)
        objs = evaluation(args, sample)
        sample.objs = objs

        sample_batch.append(sample)
        scalarization_batch.append(scalarization)
    
    temp_env.close()

    return sample_batch, scalarization_batch


def run(args):
    """
    Runs the entire MORL algorithm. (See Alg. 1 in paper)
    """
    # Torch stuff
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # Initialization
    scalarization_template = WeightedSumScalarization(num_objs = args.obj_num, weights = np.ones(args.obj_num) / args.obj_num)
    total_num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    start_time = time.time()
    
    external_pareto = EP()
    population = Population()
    opt_graph = OptGraph()
    
    selected_tasks, scalarization_batch = initialize_warmup_batch(args, device)
    rl_num_updates = args.warmup_iter
    for sample, scalarization in zip(selected_tasks, scalarization_batch):
        sample.optgraph_id = opt_graph.insert(deepcopy(scalarization.weights), deepcopy(sample.objs), -1)

    episode = 0
    iteration = 0
    while iteration < total_num_updates:
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

        # Parallel computation for MOPG
        processes = []
        results_queue = Queue()
        finished_event = Event()
    
        for task_id, task in enumerate(task_batch):
            worker_args = (args, task_id, task, device, iteration, rl_num_updates, start_time, results_queue, done_event)
            worker = Process(target=mopg_worker, args=worker_args)
            worker.start()
            processes.append(worker)

        # collect MOPG results for offsprings and insert objs into objs buffer
        all_offspring_batch = [[] for _ in range(len(processes))]
        cnt_done_workers = 0
        while cnt_done_workers < len(processes):
            rl_results = results_queue.get()
            task_id, offsprings = rl_results['task_id'], rl_results['offspring_batch']
            for sample in offsprings:
                all_offspring_batch[task_id].append(Sample.copy_from(sample))
            if rl_results['done']:
                cnt_done_workers += 1
        
        # put all intermediate policies into all_sample_batch for EP update
        all_sample_batch = [] 
        # store the last policy for each optimization weight for RA
        last_offspring_batch = [None] * len(processes) 
        # only the policies with iteration % update_iter = 0 are inserted into offspring_batch for population update
        # after warm-up stage, it's equivalent to the last_offspring_batch
        offspring_batch = [] 
        for task_id in range(len(processes)):
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

        finished_event.set()

        # ----------------------> Update EP <------------------------ #
        ep.update(all_sample_batch)
        population.update(offspring_batch)

        # ------------------- > Task Selection <--------------------- #

        selected_tasks, scalarization_batch, predicted_offspring_objs = \
            population.prediction_guided_selection(args, iteration, ep, opt_graph, scalarization_template)

        print_info('Selected Tasks:')
        for i in range(len(selected_tasks)):
            print_info('objs = {}, weight = {}'.format(selected_tasks[i].objs, scalarization_batch[i].weights))

        iteration = min(iteration + rl_num_updates, total_num_updates)

        rl_num_updates = args.update_iter

        # ----------------------> Save Results <---------------------- #
        # save ep
        ep_dir = os.path.join(args.save_dir, str(iteration), 'ep')
        os.makedirs(ep_dir, exist_ok = True)
        with open(os.path.join(ep_dir, 'objs.txt'), 'w') as fp:
            for obj in ep.obj_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*obj))

        # save population
        population_dir = os.path.join(args.save_dir, str(iteration), 'population')
        os.makedirs(population_dir, exist_ok = True)
        with open(os.path.join(population_dir, 'objs.txt'), 'w') as fp:
            for sample in population.sample_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(sample.objs)))
        # save optgraph and node id for each sample in population
        with open(os.path.join(population_dir, 'optgraph.txt'), 'w') as fp:
            fp.write('{}\n'.format(len(opt_graph.objs)))
            for i in range(len(opt_graph.objs)):
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + ';{:5f}' + (args.obj_num - 1) * ',{:5f}' + ';{}\n')\
                         .format(*(opt_graph.weights[i]), *(opt_graph.objs[i]), opt_graph.prev[i]))
            fp.write('{}\n'.format(len(population.sample_batch)))
            for sample in population.sample_batch:
                fp.write('{}\n'.format(sample.optgraph_id))

        # save selected tasks
        elite_dir = os.path.join(args.save_dir, str(iteration), 'elites')
        os.makedirs(elite_dir, exist_ok = True)
        with open(os.path.join(elite_dir, 'elites.txt'), 'w') as fp:
            for elite in selected_tasks:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(elite.objs)))
        with open(os.path.join(elite_dir, 'weights.txt'), 'w') as fp:
            for scalarization in scalarization_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(scalarization.weights)))
        if args.selection_method == 'prediction-guided':
            with open(os.path.join(elite_dir, 'predictions.txt'), 'w') as fp:
                for objs in predicted_offspring_objs:
                    fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(objs)))
        with open(os.path.join(elite_dir, 'offsprings.txt'), 'w') as fp:
            for i in range(len(all_offspring_batch)):
                for j in range(len(all_offspring_batch[i])):
                    fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(all_offspring_batch[i][j].objs)))

    # ----------------------> Save Final Model <---------------------- 

    os.makedirs(os.path.join(args.save_dir, 'final'), exist_ok = True)

    # save ep policies
    for i, sample in enumerate(ep.sample_batch):
        torch.save(sample.actor_critic.state_dict(), os.path.join(args.save_dir, 'final', 'EP_policy_{}.pt'.format(i)))
    
    # save all ep objectives
    with open(os.path.join(args.save_dir, 'final', 'objs.txt'), 'w') as fp:
        for i, obj in enumerate(ep.obj_batch):
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(obj)))

    # # save all ep env_params
    # if args.obj_rms:
    #     with open(os.path.join(args.save_dir, 'final', 'env_params.txt'), 'w') as fp:
    #         for sample in ep.sample_batch:
    #             fp.write('obj_rms: mean: {} var: {}\n'.format(sample.env_params['obj_rms'].mean, sample.env_params['obj_rms'].var))

