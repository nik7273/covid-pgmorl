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
from mopg import evaluation
from population import Population
from opt_graph import OptGraph

import torch.optim as optim

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
    for sample, scalarization in zip(elite_batch, scalarization_batch):
        sample.optgraph_id = opt_graph.insert(deepcopy(scalarization.weights), deepcopy(sample.objs), -1)
    
    # Warm-up


    # Evolution


    # Policy choice 
