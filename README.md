Multi-Objective Reinforcement Learning for Optimal COVID-19 Control in Michigan
==================
*This is not a peer-reviewed project, nor is it published.*  
Final Project for EECS 598: Reinforcement Learning Theory @ University of Michigan, Ann Arbor  

**Authors (Departments)** (alphabetical):
+ Alex Chen (Electrical and Computer Engineering)  
+ Andy Chen (Physics)  
+ Nikhil Devraj (Computer Science and Engineering)  
+ Bowei Li (Civil and Environmental Engineering)  
+ Daniel Otero-Leon (Industrial Operations Engineering)  
+ Nisarg Trivedi (Electrical and Computer Engineering)  

**Abstract**:  
COVID-19 has ravaged the world over the past year, and governments worldwide have struggled to stop the pandemic from spreading and presenting debilitating societal and economic effects. They wield the power to control major operations that could help minimize losses, such as the ability to impose lockdowns and distribute vaccines. However, there exist trade-offs between balancing economic and societal costs, calling for the design of policies that can balance these objectives as best as possible. In this project, we investigate how multi-objective reinforcement learning can help devise such Pareto-optimal control policies. We find that our policies have many different resulting strategies for dealing with the virus in the state of Michigan. Based on our findings, the approach in question proves to be promising for considering different ways to handle pandemics and how they may perform with respects to the economy and public health.  

## Notes
The repository in its current state has not been cleaned. As a result, much of it is currently not executable, since we ran our code and produced results primarily in Colab notebooks (excluded for security reasons). 

There were two primary RL approaches we implemented for policy optimization - a Deep Double Q-Network Approach and a baseline variant on Value Iteration. The repository has been split based on these approaches. You will see repeated code because these were worked on in parallel.   

## Repository Tree
This repository contains code used to generate results for our course project. 
We show the structure of the repository here with short descriptions:  
```
Repository Tree
covid-pgmorl/
# DDQN Approach
├── ddqn
│   └── src
│       ├── class_defs.py
│       ├── comparison.py
│       ├── DDQN.py
│       ├── imports.py
│       ├── model_cal.py
│       ├── mopg.py
│       ├── morl.py
│       ├── plots.py
│       ├── population.py
│       ├── run.py
│       ├── SIR.py
│       ├── util_func.py
│       └── warmup.py
├── README.md
# Value Iteration Approach
└── value-iteration
    ├── datasets
    │   └── np_arr.pt
    ├── src
    │   ├── arguments.py
    │   ├── class_defs.py
    │   ├── mopg.py
    │   ├── mopo.py
    │   ├── morl.py
    │   ├── output.py
    │   ├── plots.py
    │   ├── population.py
    │   ├── run.py
    │   ├── sir_model_env.py
    │   └── utils.py
    # Testing the GSIR environment
    ├── tests
    │   ├── fulllockdown.conf
    │   ├── fulllockdown_deterministic.npy
    │   ├── fulllockdown_stochastic_0.npy
    │   ├── fulllockdown_stochastic_1.npy
    │   ├── ...
    # Trained GSIR models
    ├── trained_models
    └── __utils
        ├── gen_toys.py
        ├── model_calibration.py
        └── sir_environment.py
```

## Acknowledgements
We thank [Xu et al.](https://github.com/mit-gfx/PGMORL) for heavy inspiration. Some of our code was adapted from their repository to suit our needs, since we employed their proposed PGMORL algorithm in order to generate our recommended policies.  
We also thank the authors of the [Michigan.gov Coronavirus page](https://www.michigan.gov/coronavirus/0,9753,7-406-98163_98173---,00.html) for making their COVID data publicly accessible.  
We finally thank Dr. Lei Ying for organizing and teaching the Reinforcement Learning Theory course here at the University of Michigan.  

## Inquiries and Concerns
Any inquiries and/or concerns about this repository should be directed to devrajn (at) umich (dot) edu.  
