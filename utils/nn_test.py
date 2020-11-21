from sir_environment import SIR_env
from policy_template import PolicyTemplate, DDQN
import numpy as np


beta            = [0.12363423292230269, 0.10176763955434223]
gamma           = 0.10622938482036372
M               = 9986857
nsteps          = 100

X_S0            = 9986324
X_I0            = 519
X_R0            = 14

'''
f_R0 = np.random.rand()
f_I0 = np.random.rand()*(1-f_R0)
f_S0 = 1 - f_R0 - f_I0
'''
f_S0 = X_S0 / M
f_I0 = X_I0 / M
f_R0 = X_R0 / M

sir_env = SIR_env()

sir_env.setup(f_S0*M, f_I0*M, f_R0*M, beta, gamma, M)
sir_nn = DDQN()

for ii in range(nsteps):
  action = sir_nn.select_action([f_R0, f_I0])
  sir_env.timeStep(action)
  X_S0, X_I0, X_R0 = sir_env.getDeterministic()

  # TODO: calculate reward


  f_R0 = X_R0 / M
  f_I0 = X_I0 / M

  if ((ii % 10) == 0):
    print("%i of %i steps." % (ii+1, nsteps))

X_S = sir_env.X_S
X_I = sir_env.X_I
X_R = sir_env.X_R

print("X_S: ")
print(X_S)
print("X_I: ")
print(X_I)
print("X_R: ")
print(X_R)
