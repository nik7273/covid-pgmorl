import numpy as np
from utils import *
from abc import abstractmethod
import torch

"""
Scalarization Methods
"""
class ScalarizationFunction():
    def __init__(self, num_objs, weights = None):
        self.num_objs = num_objs
        if weights is not None:
            self.weights = torch.Tensor(weights)
        else:
            self.weights = None
    
    def update_weights(self, weights):
        if weights is not None:
            self.weights = torch.Tensor(weights)

    @abstractmethod
    def evaluate(self, objs):
        pass

class WeightedSumScalarization(ScalarizationFunction):
    def __init__(self, num_objs, weights = None):
        super(WeightedSumScalarization, self).__init__(num_objs, weights)
    
    def update_z(self, z):
        pass

    def evaluate(self, objs):
        return (objs * self.weights).sum(axis = -1)

'''
Define a external pareto class storing all computed policies on the current pareto front.
'''
class EP:
    def __init__(self):
        self.obj_batch = np.array([])
        self.sample_batch = np.array([])

    def index(self, indices, inplace=True):
        if inplace:
            self.obj_batch, self.sample_batch = \
                map(lambda batch: batch[np.array(indices, dtype=int)], [self.obj_batch, self.sample_batch])
        else:
            return map(lambda batch: deepcopy(batch[np.array(indices, dtype=int)]), [self.obj_batch, self.sample_batch])

    def update(self, sample_batch):
        self.sample_batch = np.append(self.sample_batch, np.array(deepcopy(sample_batch)))
        for sample in sample_batch:
            self.obj_batch = np.vstack([self.obj_batch, sample.objs]) if len(self.obj_batch) > 0 else np.array([sample.objs])

        if len(self.obj_batch) == 0: return
        ep_indices = get_ep_indices(self.obj_batch)

        self.index(ep_indices)

'''
OptGraph is a data structure to store the optimization history.
The optimization history is a rooted forest, and is organized in a tree structure.
'''
class OptGraph:
    def __init__(self):
        self.weights = []
        self.objs = []
        self.delta_objs = []
        self.prev = []
        self.succ = []

    def insert(self, weights, objs, prev):
        self.weights.append(deepcopy(weights) / np.linalg.norm(weights))
        self.objs.append(deepcopy(objs))
        self.prev.append(prev)
        if prev == -1:
            self.delta_objs.append(np.zeros_like(objs))
        else:
            self.delta_objs.append(objs - self.objs[prev])
        if prev != -1:
            self.succ[prev].append(len(self.objs) - 1)
        self.succ.append([])
        return len(self.objs) - 1


"""
Define a MOPG task, which is a pair of a policy and a scalarization weight.
"""
class Task:
    def __init__(self, sample, scalarization):
        self.sample = Sample.copy_from(sample)
        self.scalarization = deepcopy(scalarization)
        
'''
Sample is a policy with associated environment variables.
'''
class Sample:
    def __init__(self, objs = None, optgraph_id = None):
        self.objs = objs
        self.X_I = 0
        self.X_S = 0 
        self.val_I = []
        self.val_L = []
        self.policy = -1
        
        self.optgraph_id = optgraph_id

    @classmethod
    def copy_from(cls, sample):
        objs = deepcopy(sample.objs)
        optgraph_id = sample.optgraph_id
        return cls(agent, objs, optgraph_id)