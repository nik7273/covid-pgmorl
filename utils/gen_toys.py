#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2020-11-07

@author: andyzfchen
@email: chen.andy14@gmail.com, andych@umich.edu
"""

import numpy as np
import configparser
if __name__ == "__main__":
  import sir_environment
else:
  from .utils import sir_environment

class GenToys(object):
  def __init__(self, sir_env):
    '''
    Toy episode generator.

    Note: Config file is typically used when script is run as main. Otherwise use setup method to incorporate SIR environment from main script.
    '''
    self.config = None

  def loadConfig(self, filename):
    '''
    Loads a configuration file for generating toys.

    Input:
    - filename:   path of config file
    '''
    self.config = configparser.ConfigParser()
    self.config.read(filename)

    self.X_S0 = int(self.config["SIR Env"]["X_S0"])
    self.X_I0 = int(self.config["SIR Env"]["X_I0"])
    self.X_R0 = int(self.config["SIR Env"]["X_R0"])
    self.beta = list(map(float, self.config["SIR Env"]["beta"].split(":")))
    self.gamma = float(self.config["SIR Env"]["gamma"])
    self.M = int(self.config["SIR Env"]["M"])
    action = list(map(int, self.config["Policy"]["action"].split(":")))
    freq = list(map(int, self.config["Policy"]["freq"].split(":")))
    try:
      self.savefilename = self.config["File"]["savefilename"]
    except:
      pass

    self.actions = np.array([]).astype(int)
    for ii, jj in zip(action, freq):
      self.actions = np.append(self.actions, (np.ones(jj)*ii).astype(int))


  def setup(self, sir_env=None):
    '''
    Setup the model calibration and SIR environment.

    Input:
    - sir_env:    SIR environment
    '''
    if sir_env is None:
      print("Creating SIR environment based on config file parameters.")
      assert self.config is not None


      self.sir_env = sir_environment.SIR_env()
      self.sir_env.setup(X_S0=self.X_S0, X_I0=self.X_I0, X_R0=self.X_R0, beta=self.beta, gamma=self.gamma, M=self.M)


    else:
      print("Using SIR environment passed as argument.")
      self.sir_env = sir_env

    
  def genDeterministicToy(self, actions=None, savefile=False):
    '''
    Generates a deterministic toy episode based on parameters from the SIR environment.

    Input:
    - actions:    taken actions of episode
    '''
    toy = []
    if actions is None:
      assert self.sir_env is not None

      for ii in self.actions:
        self.sir_env.timeStep(ii)

      toy = np.array([self.sir_env.X_S_true, self.sir_env.X_I_true, self.sir_env.X_R_true])
    else:
      for ii in actions:
        self.sir_env.timeStep(ii)

      toy = np.array([self.sir_env.X_S_true, self.sir_env.X_I_true, self.sir_env.X_R_true])

    if savefile:
      np.save("%s_deterministic.npy" % self.savefilename, toy)
    return toy
    

  def genStochasticToy(self, actions=None, n=1, savefile=False):
    '''
    Generates a stochastic toy episode based on parameters from the SIR environment.

    Input:
    - actions:    taken actions of episode
    - n:          number of stochastic episodes
    '''
    assert self.sir_env is not None

    if actions is None:
      actions = self.actions

    X_S = np.zeros((n, len(actions))).astype(int)
    X_I = np.zeros((n, len(actions))).astype(int)
    X_R = np.zeros((n, len(actions))).astype(int)

    X_S[:,0] = self.X_S0
    X_I[:,0] = self.X_I0
    X_R[:,0] = self.X_R0

    for ii in range(1,len(actions)):
      self.sir_env.timeStep(actions[ii])
      for jj in range(n):
        X_S[jj,ii] , X_I[jj,ii], X_R[jj,ii] = self.sir_env.sampleStochastic(last_X_S=X_S[jj,ii-1], last_X_R=X_R[jj,ii-1])


    if savefile:
      for jj in range(n):
        toy = np.array([X_S[jj,:], X_I[jj,:], X_R[jj,:]])
        np.save("%s_stochastic_%i.npy" % (self.savefilename, jj), toy)

    return X_S, X_I, X_R




def main():
  '''
  Main method when executed as a script
  '''
  from argparse import ArgumentParser

  parser = ArgumentParser(description='Generate some episodes.')
  parser.add_argument('filename', type=str, help='Directory path for configuration file.')
  args = parser.parse_args()

  sir_env = sir_environment.SIR_env()
  gt = GenToys(sir_env)
  gt.loadConfig(args.filename)
  gt.setup()
  deterministictoy = gt.genDeterministicToy(savefile=True)
  print(deterministictoy)
  gt.setup()
  stochastictoy = gt.genStochasticToy(n=10, savefile=True)
  print(stochastictoy)
  #print(np.all((stochastictoy[0][0,:-1] - stochastictoy[0][0,1:]) >= 0))
  #print(np.all((stochastictoy[2][0,1:] - stochastictoy[2][0,:-1]) >= 0))
  




if __name__ == "__main__":
  main()
