#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2020-11-07

@author:  Bowei Li
@email:   jacklbw@umich.edu 
"""

import numpy as np
import matplotlib.pyplot as plt

class model_calibration(object):
    def __init__(self,O_I,ac,M = 9986857, D = 9):
      # Inputs: (Assume that all quantities are with consistent units)
      # 1. Observed cumulative confirmed cases.
      # 3. Action time series, ac (an n_t by 1 column vector)
      # 4. Total population M, a constant number (we ignore the population change due to natural born or death); M = 9986857 is Total population as per July,2019 Census data. 
      # 5. Mean time delay of a infected person be confirmed. D = 9, suggested by Sun et al. 2020, Pellis et al. 2020
      self.X_R = O_I[0:-D] # Removal population time series, X_R (an n_t by 1 column vector)
      self.X_I = O_I[D:]-O_I[0:-D] # Infected population time series, X_I (an n_t by 1 column vector)
      self.ac = ac[0:-D]
      self.M = M
      self.X_S = self.M - self.X_R - self.X_I
      self.D = D # Infected to removal delay

      # added by Andy Chen for the step class that inherits this class (change as needed)
      self.beta = 0
      self.gamma = 0

    def model_mls(self):
      # Outputs:
      # model parameters beta, gamma
      # Method: Maximum likelihood Estimation (Report section 2.2)

      # Beta
      Z_S = self.X_S[0:-1] * self.X_I[0:-1] / self.M
      Y_S = self.X_S[0:-1] - self.X_S[1:]
      ac_list = np.unique(self.ac)
      beta = [0]*len(ac_list) # a list of zeros of same length as 'ac_list'.
      # print(self.X_S[:5], self.X_I[:5], self.X_R[:5])
      # print(np.array([np.zeros(self.D,1),self.ac[0:-self.D]]))
      # print(self.ac[0:-self.D])
      # ac_shift = np.concatenate((np.zeros((self.D,1)), self.ac[0:-self.D]))
      # ac_shift = np.array([np.zeros((1,self.D)),self.ac[0:-self.D]])
      # ac_shift = np.hstack(   (np.zeros((1,self.D)),   self.ac[0:-self.D]))
      # ac_shift = np.pad(self.ac[0:-self.D], (self.D, 0), 'edge')
      # print(ac_shift)
      for index, ac in enumerate(ac_list):
          ind_ac = np.where(self.ac[0:-1] == ac) # data index
          # ind_ac = np.where(ac_shift[0:-1] == ac) # data index
          beta[index] = np.sum(Y_S[ind_ac]) / np.sum(Z_S[ind_ac]) 
      
      # Gamma
      Z_R = self.X_I[0:-1]
      Y_R = self.X_R[1:]-self.X_R[0:-1]
      gamma = np.sum(Y_R) / np.sum(Z_R)
      # print(sum(Z_R==0))

      # added by Andy Chen for the step class that inherits this class (change as needed)
      self.beta = beta
      self.gamma = gamma
      return beta, gamma

    def model_validate(self,beta,gamma):
      # Simulate data and compare
      Nsim = 1000 # Number of simulations
      T = len(self.X_R) # Horizon

      X_R_s = np.zeros((Nsim,T))
      X_I_s = np.zeros((Nsim,T))
      X_S_s = np.zeros((Nsim,T))
      ac_list = np.unique(self.ac)
      # ac_shift = np.pad(self.ac[0:-self.D], (self.D, 0), 'edge')

      for mc in range(Nsim): # Loop over all samples 

          # Initial conditions
          X_R_s[mc][0] = self.X_R[0]
          X_I_s[mc][0] = self.X_I[0]
          X_S_s[mc][0] = self.X_S[0]

          for t in range(1, T): # Loop over all time steps

              # Current beta (depends on the action of the current step)
              ind_beta = np.squeeze(np.where(ac_list == self.ac[t]))
              # ind_beta = np.squeeze(np.where(ac_list == ac_shift[t]))
              beta_t = beta[ind_beta]

              # # The GSIR model (free-run)
              # e_S = np.random.poisson(beta_t*X_S_s[mc][t-1]*X_I_s[mc][t-1]/self.M)
              # e_R = np.random.binomial(X_I_s[mc][t-1], gamma)
              # X_S_s[mc][t] = X_S_s[mc][t-1] - e_S
              # X_R_s[mc][t] = X_R_s[mc][t-1] + e_R
              # X_I_s[mc][t] = self.M - X_S_s[mc][t] - X_R_s[mc][t]
              
              # The GSIR model (One-step-ahead)
              e_S = np.random.poisson(beta_t*self.X_S[t-1]*self.X_I[t-1]/self.M)
              e_R = np.random.binomial(self.X_I[t-1], gamma)
              X_S_s[mc][t] = X_S_s[mc][t-1] - e_S
              X_R_s[mc][t] = X_R_s[mc][t-1] + e_R
              X_I_s[mc][t] = self.M - X_S_s[mc][t] - X_R_s[mc][t]
      
      t = range(T)
      print('Red: Observation, Gray: Prediction samples')
      print('Removal population (confirmed cases)')
      h1=plt.plot(t, self.X_R, color='r', linewidth=2.0)
      for mc in range(Nsim):
          h2=plt.plot(t, X_R_s[mc][:], color=[0.5, 0.5, 0.5], linewidth=0.5)
      plt.title('Removal population')
      # plt.legend((h1,h2),('Observation','Prediction'))
      plt.show()

      print('Infected population')
      plt.plot(t, self.X_I, color='r', linewidth=2.0)
      for mc in range(Nsim):
          plt.plot(t, X_I_s[mc][:], color=[0.5, 0.5, 0.5], linewidth=0.5)
      plt.title('Infected population')
      # plt.legend((h1,h2),('Observation','Prediction'))
      plt.show()

      print('Susceptible population')
      plt.plot(t, self.X_S, color='r', linewidth=2.0)
      for mc in range(Nsim):
          plt.plot(t, X_S_s[mc][:], color=[0.5, 0.5, 0.5], linewidth=0.5)
      plt.title('Susceptible population')
      # plt.legend((h1,h2),('Observation','Prediction'))
      plt.show()

