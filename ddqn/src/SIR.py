"""
GSIR simulator environment.
"""

class SIR_env(object):
  def __init__(self, model_calibration=None):
    '''
    SIR population dynamics model environment that allows user to evolve through time given action parameters.
    Input:
    - model_calibration:      SIR model calibration to be associated to this object and where hyperparameters will be drawn
    '''
    if model_calibration is not None:
      #print("Using model calibration object passed as argument.")
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
      self.rewards = np.array([])

      self.beta = self.model_calibration.beta
      self.gamma = self.model_calibration.gamma
      self.M = self.model_calibration.M
      # print("Beta: ", self.beta)
      # print("Gamma: ", self.gamma)
      # print("M: ", self.M)
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
    self.X_S = np.array([self.model_calibration.X_S[0]]).astype(int)
    self.X_I = np.array([self.model_calibration.X_I[0]]).astype(int)
    self.X_R = np.array([self.model_calibration.X_R[0]]).astype(int)
    self.std_X_S = np.zeros(1)
    self.std_X_I = np.zeros(1)
    self.std_X_R = np.zeros(1)
    self.actions = np.array([])
    self.rewards = np.array([])

    state = np.array([self.X_S_true[0], self.X_I_true[0], self.X_R_true[0]])  ### BIG BIG BUG CAUGHT!! It was all X_S_true[0]. We found it!!!
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
    #print(self.beta)
    #print(self.X_S_true[-1])
    #print(self.X_I_true[-1])
    #print(self.beta[action]*self.X_S_true[-1]*self.X_I_true[-1]/self.M)

    #DS
    mean_e_I_new = int(np.floor(self.beta[action]*self.X_S_true[-1]*self.X_I_true[-1]/self.M))
    #mean_e_I_new = int(np.floor(self.beta*self.X_S_true[-1]*self.X_I_true[-1]/self.M))
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

    self.rewards = np.append(self.rewards, reward)
    done = 0
    time_horizon = 50
    if (len(self.rewards)/2 >= time_horizon): # len(self.rewards) is 2*real time horizon, because reward has 2 elements in it.
      done = -1
    elif (self.X_S[-1] < self.M/2):
      done = 1
    info = {"obj": reward}

    return next_state, reward, done, info

  def getReward(self):
    # societal cost: 
    # TODO: calculated based on true values; maybe consider something with the stochastic data later

    ### Bowei: Dec.04
    X_S = self.X_S  # We should use X_S_true?????
    X_I = self.X_I
    ### Bowei

    #cost_s = self.X_I_true[-1] / (self.X_S_true[-1] + self.X_I_true[-1])
    #cost_s = self.X_I_truth[-1] + self.X_R_truth[-1] - (self.X_I_truth[-2] + self.X_R_truth[-2])
    
    # cost_s = (self.X_I_true[-1] + self.X_R_true[-1]) - (self.X_I_true[-2] + self.X_R_true[-2])
    # cost_s = (self.X_S_true[-2] - self.X_S_true[-1]) / self.X_S_true[-2]
    cost_s = self.X_S_true[-2] - self.X_S_true[-1]
    """
    # societal cost now also has a future value calculated based on possible future infected and removed
    meanX_S, meanX_I, meanX_R = self.getDeterministic()
    
    ### Bowei: Dec.04
    # policy = self.actions[-1] 
    policy = int(self.actions[-1]) 
    ### Bowei

    errX_S, errX_I, errX_R = self.getError()

    # 1 sigma bound for X_S
    lowXS = max(round(meanX_S - errX_S, 0), 0)
    uppXS = min(round(meanX_S + errX_S, 0), self.M)
    
    # 1 sigma bound for X_R
    lowXR = max(round(meanX_R - errX_R, 0), 0)
    uppXR = min(round(meanX_R + errX_R, 0), self.M)
    
    # bounds for possible new infected
    lowI = int(max(X_S[-1] - uppXS, 0))
    uppI = int(min(X_S[-1], X_S[-1] - lowXS))
    
    # bounds for possible new removed
    lowR = int(max(lowXR - (self.M - X_I[-1] - X_S[-1]), 0))
    uppR = int(min(X_I[-1], uppXR - (self.M - X_I[-1] - X_S[-1])))

    for i in range(lowI,uppI):
      #for j in range(lowR,uppR):
      probI = scipy.stats.poisson.pmf(i, self.beta[policy])
      #probR=binom.pmf(j,uppR,env.gamma)
      
      if i==lowI:
          probI = scipy.stats.poisson.cdf(i, self.beta[policy])
          
      if i==uppI:
          probI = 1 - scipy.stats.poisson.cdf(i-1, self.beta[policy])
      
      #if j==lowR:
      #    probR=binom.cdf(j,uppR,env.gamma)
      #if j==uppR:
      #    probR=1-binom.cdf(j-1,uppR,env.gamma)
      
      
      #val_I+=0.97*probI*probR*currentV_I[X_I+i-j-1,X_S-i-1]
      #val_L+=0.97*probI*probR*currentV_L[X_I+i-j-1,X_S-i-1]
      cost_s += probI*i
    """
    # economic cost: 
    r_ld = 1 # TODO: incorporate a more robust lockdown cost rate
    #this doesn't make sense but just testing
    #r_ld = (random.random() *0.2 - 0.1) + 1.0 # TODO: incorporate a more robust lockdown cost rate
    #add small value to avoid being zero
    cost_e = ((self.actions[-1] + 1e-8) * r_ld) #+ (random.random() *0.4 - 0.2))
    # Cost to consider changing actions
    #cost_e = (self.actions[-1] + 1e-6) * r_ld + 0.5 * abs(self.actions[-1] - self.actions[-2])
    # cost = np.array([(cost_s - 2500)/2500, cost_e /50]) #* 1e6 # Dec.06 Normalized to be in [0, 1]
    cost = np.array([-1.0*cost_s, -1.0*cost_e]) # Will do normalization in reward_normal
    #[0.00020717541300372262, 4000.00004] 

    return cost

  def getDeterministic(self):
    '''
    Retrieve the most recently calculated X_S, X_I, and X_R mean values (as if the model was deterministic).
    Output:s
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
