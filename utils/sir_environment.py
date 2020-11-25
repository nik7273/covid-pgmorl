#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2020-11-07

@author:  Andy Chen
@email:   chen.andy14@gmail.com, andych@umich.edu
"""

import numpy as np

class SIR_env(object):
  def __init__(self, model_calibration=None):
    '''
    SIR population dynamics model environment that allows user to evolve through time given action parameters.

    Input:
    - model_calibration:      SIR model calibration to be associated to this object and where hyperparameters will be drawn
    '''
    if model_calibration is not None:
      print("Using model calibration object passed as argument.")
      self.model_calibration = model_calibration

      self.X_S_true = np.array([self.model_calibration.X_S[0]]).astype(int)
      self.X_I_true = np.array([self.model_calibration.X_I[0]]).astype(int)
      self.X_R_true = np.array([self.model_calibration.X_R[0]]).astype(int)
      self.X_S = np.array([self.model_calibration.X_S[0]]).astype(int)
      self.X_I = np.array([self.model_calibration.X_I[0]]).astype(int)
      self.X_R = np.array([self.model_calibration.X_R[0]]).astype(int)
      self.std_X_S = np.zeros(1)
      self.std_X_I = np.zeros(1)
      self.std_X_R = np.zeros(1)
      self.actions = np.array([])

      self.beta = self.model_calibration.beta
      self.gamma = self.model_calibration.gamma
      self.M = self.model_calibration.M
      print("Beta: ", self.beta)
      print("Gamma: ", self.gamma)
      print("M: ", self.M)
    else:
      print("No model calibration object passed.")
      self.X_S_true = np.array([])
      self.X_I_true = np.array([])
      self.X_R_true = np.array([])
      self.X_S = np.array([])
      self.X_I = np.array([])
      self.X_R = np.array([])
      self.std_X_S = np.zeros(1)
      self.std_X_I = np.zeros(1)
      self.std_X_R = np.zeros(1)

      self.beta = None
      self.gamma = None
      self.M = None
      print("Use setup method to define model parameters.")

  def reset(self):
    self.X_S_true = np.array([self.X_S_true[0]])
    self.X_I_true = np.array([self.X_I_true[0]])
    self.X_R_true = np.array([self.X_R_true[0]])

    state = [self.X_S_true[0], self.X_S_true[0], self.X_S_true[0]]
    done = False

    return state, done

  def setup(self, X_S0=None, X_I0=None, X_R0=None, beta=None, gamma=None, M=None):
    if X_S0 is not None:
      self.X_S_true = np.array([X_S0]).astype(int)
      self.X_S = np.array([X_S0]).astype(int)
    if X_I0 is not None:
      self.X_I_true = np.array([X_I0]).astype(int)
      self.X_I = np.array([X_I0]).astype(int)
    if X_R0 is not None:
      self.X_R_true = np.array([X_R0]).astype(int)
      self.X_R = np.array([X_R0]).astype(int)
    if beta is not None:
      self.beta = beta
    if gamma is not None:
      self.gamma = gamma
    if M is not None:
      self.M = M

  def setSeed(self, seed):
    '''
    Defines the seed for reproducibility in the stochastic model.

    Input:
    - seed:                   random seed to use when using the step functions
    '''
    np.random.seed(seed)

  def timeStep(self, action):
    '''
    Evolve the model through one time step given an action.

    Input:
    - action:                 action to use when evolving the model by one time step
    '''
    assert self.beta is not None
    assert self.gamma is not None
    assert self.M is not None

    # mean population transition from X_S to X_I
    mean_e_I_new = int(np.floor(self.beta[action]*self.X_S_true[-1]*self.X_I_true[-1]/self.M))
    # mean population transition from X_I to X_R
    mean_e_R_new = int(np.floor(self.gamma*self.X_I_true[-1]))

    # new mean populations
    X_S_new = self.X_S_true[-1] - mean_e_I_new
    X_I_new = self.X_I_true[-1] + mean_e_I_new - mean_e_R_new
    X_R_new = self.X_R_true[-1] + mean_e_R_new

    # error propagation due to stochastic model
    std_X_S_new = np.sqrt(self.beta[action]*self.X_S_true[-1]*self.X_I_true[-1]/self.M)
    std_X_R_new = self.X_I_true[-1] * (self.gamma)*(1-self.gamma)
    std_X_I_new = np.sqrt(std_X_S_new**2 + std_X_R_new**2)

    # update X lists and std lists
    self.X_S_true = np.append(self.X_S_true, X_S_new)
    self.X_I_true = np.append(self.X_I_true, X_I_new)
    self.X_R_true = np.append(self.X_R_true, X_R_new)

    # update actions list
    self.actions = np.append(self.actions, action)

    # ensure X_S is monotonically decreasing or equal
    temp_X_S = int(np.random.normal(self.X_S_true[-1],self.std_X_S[-1]))
    while (temp_X_S > self.X_S[-1]):
      temp_X_S = int(np.random.normal(self.X_S_true[-1],self.std_X_S[-1]))
    self.X_S = np.append(self.X_S, temp_X_S)
    # ensure X_I >= 0 and X_R is monotonically increasing or equal
    temp_X_R = int(np.random.normal(self.X_R_true[-1],self.std_X_R[-1]))
    while (((self.M - temp_X_S - temp_X_R) < 0) or (temp_X_R < self.X_R[-1])):
      temp_X_R = int(np.random.normal(self.X_R_true[-1],self.std_X_R[-1]))
    self.X_R = np.append(self.X_R, temp_X_R)
    self.X_I = np.append(self.X_I, self.M - self.X_S[-1] - self.X_R[-1])

    self.std_X_S = np.append(self.std_X_S, np.sqrt(self.std_X_S[-1]**2 + std_X_S_new**2))
    self.std_X_I = np.append(self.std_X_I, np.sqrt(self.std_X_I[-1]**2 + std_X_I_new**2))
    self.std_X_R = np.append(self.std_X_R, np.sqrt(self.std_X_R[-1]**2 + std_X_R_new**2))

    next_state = self.getDeterministic()
    reward = self.getReward()
    done = 0
    if (len(self.rewards) >= 7):
      done = -1
    elif (self.X_S < self.M/2):
      done = 1
    info = {"obj": reward}

    return next_state, reward, done, info

  def getReward(self):
    # societal cost: 
    # TODO: calculated based on true values; maybe consider something with the stochastic data later
    cost_s = self.X_I_true[-1] / (self.X_S_true[-1] + self.X_I_true[-1])

    # economic cost: 
    r_ld = 1.0  # TODO: incorporate a more robust lockdown cost rate
    cost_e = self.actions[-1] * r_ld

    cost = [cost_s, cost_e]
    return cost

  def getDeterministic(self):
    '''
    Retrieve the most recently calculated X_S, X_I, and X_R mean values (as if the model was deterministic).

    Output:
    - X_S_true:               the most recently calculated X_S
    - X_I_true:               the most recently calculated X_I
    - X_R_true:               the most recently calculated X_R

    '''
    return self.X_S_true[-1], self.X_I_true[-1], self.X_R_true[-1]

  def getError(self):
    '''
    Retrieve the most recently calculated X_S, X_I, and X_R standard deviations (using the random variable distributions of the stochastic model).

    Output:
    - std_X_S:                the most recently calculated std_X_S
    - std_X_I:                the most recently calculated std_X_I
    - std_X_R:                the most recently calculated std_X_R

    '''
    return self.std_X_S[-1], self.std_X_I[-1], self.std_X_R[-1]

  def sampleStochastic(self, last_X_S=None, last_X_R=None):
    '''
    Samples a set of the most recently calculated X_S, X_I, and X_R values (assuming CLT).

    Output:
    - X_S_samp:               the most recently calculated X_S
    - X_I_samp:               the most recently calculated X_I
    - X_R_samp:               the most recently calculated X_R

    '''
    X_S_samp = int(np.random.normal(self.X_S_true[-1],self.std_X_S[-1]))
    # ensure X_S is monotonically decreasing or equal
    if last_X_S is not None:
      while (X_S_samp > last_X_S):
        X_S_samp = int(np.random.normal(self.X_S_true[-1],self.std_X_S[-1]))
    X_R_samp = int(np.random.normal(self.X_R_true[-1],self.std_X_R[-1]))
    # ensure X_I >= 0
    if last_X_R is not None:
      # ensure X_R is monotonically increasing or equal
      while ((self.M - X_S_samp - X_R_samp) < 0) or (X_R_samp < last_X_R):
        X_R_samp = int(np.random.normal(self.X_R_true[-1],self.std_X_R[-1]))
    else:
      while ((self.M - X_S_samp - X_R_samp) < 0):
        X_R_samp = int(np.random.normal(self.X_R_true[-1],self.std_X_R[-1]))
    X_I_samp = self.M - X_S_samp - X_R_samp
    return X_S_samp, X_I_samp, X_R_samp
