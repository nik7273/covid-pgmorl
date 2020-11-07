#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2020-11-07

@author:  Andy Chen
@email:   chen.andy14@gmail.com, andych@umich.edu
"""

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

  def setup(self, beta=None, gamma=None, M=None):
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
    mean_e_I_new = np.floor(self.beta[action]*self.X_S[-1]*self.X_I[-1]/self.M)
    # mean population transition from X_I to X_R
    mean_e_R_new = np.floor(self.gamma*self.X_I[-1])

    # new mean populations
    X_S_new = self.X_S_true[-1] - mean_e_I_new
    X_I_new = self.X_I_true[-1] + mean_e_I_new - mean_e_R_new
    X_R_new = self.X_R_true[-1] + mean_e_R_new

    # error propagation due to stochastic model
    std_X_S_new = np.sqrt(self.beta[action]*self.X_S_true[-1]*self.X_I_true[-1]/self.M)
    std_X_R_new = self.X_I_true[-1] * (self.gamma)*(1-self.gamma)
    std_X_I_new = np.sqrt(std_X_S_new**2 + std_X_R_new**2)

    # update X lists and std lists
    self.X_S_true = np.append(self.X_S, X_S_new)
    self.X_I_true = np.append(self.X_I, X_I_new)
    self.X_R_true = np.append(self.X_R, X_R_new)

    self.std_X_S = np.append(self.std_X_S, np.sqrt(self.std_X_S[-1]**2 + std_X_S_new**2))
    self.std_X_I = np.append(self.std_X_I, np.sqrt(self.std_X_I[-1]**2 + std_X_I_new**2))
    self.std_X_R = np.append(self.std_X_R, np.sqrt(self.std_X_R[-1]**2 + std_X_R_new**2))

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

  def sampleStochastic(self):
    '''
    Samples a set of the most recently calculated X_S, X_I, and X_R values (assuming CLT).

    Output:
    - X_S_samp:               the most recently calculated X_S
    - X_I_samp:               the most recently calculated X_I
    - X_R_samp:               the most recently calculated X_R

    '''
    X_S_samp = np.random.normal(self.X_S_true[-1],self.std_X_S[-1])
    X_R_samp = np.random.normal(self.X_R_true[-1],self.std_X_R[-1])
    X_I_samp = self.M - X_S_samp - X_R_samp
    return X_S_samp, X_I_samp, X_R_samp
