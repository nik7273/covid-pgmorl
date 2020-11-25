import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import time
import copy

class PolicyTemplate(torch.nn.Module):
  def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
    """DQN Network example
    Args:
        input_dim (int): `state` dimension.
            `state` is 2-D tensor of shape (n, input_dim)
        output_dim (int): Number of actions.
            Q_value is 2-D tensor of shape (n, output_dim)
        hidden_dim (int): Hidden dimension in fc layer
    """
    super(PolicyTemplate, self).__init__()

    self.layer1 = torch.nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      torch.nn.ReLU()
    )

    self.layer2 = torch.nn.Sequential(
      torch.nn.Linear(hidden_dim, hidden_dim),
      torch.nn.ReLU()
    )

    self.final = torch.nn.Linear(hidden_dim, output_dim)

  def forward(self, x: torch.Tensor):
    """Returns a Q_value
    Args:
      x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
    Returns:
      torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
    """
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.final(x)

    return x

class DDQN(object):
  def __init__(self,  nstate=2, naction=2, nlayernode=16):
    self.nstate = nstate
    self.naction = naction
    self.nlayernode = nlayernode
    self.episode_len = 7
    self.iepisode = 0
    self.update_freq = 2
    self.batch_size = 64
    self.replay_buffer = 50000

    self.alpha = 0.99
    self.lr = 0.001
    self.epsilon = 1.
    self.epsilon_min = 0.1
    self.epsilon_max = 1.
    self.epsilon_decay = 0.95

    self.NN = PolicyTemplate(self.nstate, self.naction, self.nlayernode)
    self.target_NN = copy.deepcopy(self.NN)
    self.opt = optim.Adam(self.NN.parameters(), lr=self.lr)
    self.loss_fn = torch.nn.MSELoss(reduction='sum')

    self.buffer_s0 = np.zeros((self.replay_buffer, self.nstate))
    self.buffer_a0 = np.zeros(self.replay_buffer)
    self.buffer_r = np.zeros(self.replay_buffer)
    self.buffer_s1 = np.zeros((self.replay_buffer, self.nstate))
    self.buffer_done = np.zeros(self.replay_buffer)
    self.index = 0

  def select_action(self, states, batch=False):
    if batch:
      if (np.random.rand() < 1 - self.epsilon):
        action = torch.argmax(self.NN.forward(torch.from_numpy(states).float()), axis=1)
        action = action.numpy()
      else:
        action = np.floor(np.random.rand(self.batch_size)*self.naction).astype(int)
    else:
      if (np.random.rand() < 1 - self.epsilon):
        action = torch.argmax(self.NN.forward(torch.from_numpy(states).float()))
        action = action.numpy()
      else:
        action = np.floor(np.random.rand()*self.naction).astype(int)

    if (self.index >= self.batch_size):
      self.epsilon *= self.epsilon_decay
      if self.epsilon < self.epsilon_min:
        self.epsilon = self.epsilon_min
  
    return action

  def policy(self, states, batch=False):
    if batch:
      action = torch.argmax(self.target_NN.forward(torch.from_numpy(states).float()), axis=1).numpy()
    else:
      action = torch.argmax(self.target_NN.forward(torch.from_numpy(states).float())).numpy()
    return action

  def train(self,s0,a0,r,s1,sign,wts=[1.0, 0]):
    self.buffer_s0[self.index % self.replay_buffer,:] = s0
    self.buffer_a0[self.index % self.replay_buffer] = a0
    self.buffer_r[self.index % self.replay_buffer] = r
    self.buffer_s1[self.index % self.replay_buffer,:] = s1
    self.buffer_done[self.index % self.replay_buffer] = sign
    self.index += 1

    if (self.index >= self.batch_size):
      if (sign==0):
        # sample batch
        indices = np.arange(np.min([self.index, self.replay_buffer]))
        np.random.shuffle(indices)
        batch_indices = indices[:self.batch_size]

        # train online network
        self.opt.zero_grad()
        Q0 = self.NN(torch.from_numpy(self.buffer_s0[batch_indices]).float())
        Q0 = Q0.gather(1, torch.from_numpy(self.buffer_a0[batch_indices].astype(int)).view(-1,1)).view(-1)
        Q1 = self.target_NN(torch.from_numpy(self.buffer_s1[batch_indices]).float())
        a1 = self.policy(self.buffer_s0[batch_indices], batch=True)
        Q1 = Q1.gather(1, torch.from_numpy(a1).view(-1,1)).view(-1)
        loss = self.loss_fn(Q0, torch.from_numpy(self.buffer_r[batch_indices]*wts) + self.alpha*Q1*torch.from_numpy(self.buffer_done[batch_indices]==0))
        loss.backward()
        self.opt.step()
      else:
        self.iepisode += 1
        if ((self.iepisode % self.update_freq) == 0):
          self.target_NN = copy.deepcopy(self.NN)

    return
