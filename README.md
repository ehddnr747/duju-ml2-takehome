# duju-ml2-takehome
Donguk Ju's take home project for ml2 internship  


# Review
# CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING

## Summary
In the paper the authors propose an Actor-Critic, off-policy model-free algorithms based on DPG[(Deterministic Policy Gradient)](http://proceedings.mlr.press/v32/silver14.pdf) for continuous action space inspired by Deep Q-Learning[(DQN)](https://arxiv.org/abs/1312.5602).

## Introduction
- While DQN solves problems with high-dimensional observation spaces, it can only handle discrete and low-dimensional action spaces. Many tasks of interest, most notably physical control tasks, have **continuous** (real valued) and high dimensional action spaces.
- An obvious approach to adapting deep reinforcement learning methods such as DQN to continuous domains is to to simply discretize the action space. However, this has many limitations, most notably **the curse of dimensionality**: the number of actions increases exponentially with the number of degrees of freedom.
- In this work the authors present a **model-free, off-policy actor-critic algorithm** using deep function approximators that can learn policies in high-dimensional, continuous action spaces.
- A naive application of this actor-critic method with neural function approximators is **unstable** for challenging problems. Prior to DQN, it was generally believed that learning value functions using large, non-linear function approximators was difficult and unstable.
- Two innovation in this method
  1. The network is trained **off-policy** with samples from a replay buffer to **minimize correlations between samples**.
  2. The network is trained with a **target Q network** to give consistent targets during temporal difference backups.
- Proposed model-free approach which we call Deep DPG (DDPG) can learn **competitive policies for all of our tasks using low-dimensional observations (e.g. cartesian coordinates or joint angles) using the same hyper-parameters and network structure.**

## Background
- Basic Components for RL
  - Environment _E_
  - timestep _t_
  - observation _x\_t_
  - action _a\_t_
  - reward _r\_t_
  - assumption that the environment is fully observable _s_t = x_t_
  - policy _pi_ : _S -> P(A)_
  - state space _S_, action space _A_
  - initial state distribution _p(s1)_
  - transition dynamics _p(s\_(t+1)|s\_t,a\_t)_
  - return(discounted future reward) <img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/return.png" width="200">
  - discounting factor _gamma_ in [0,1]
- Goal is maximizing the expected return from the start distribution <img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/J.png" width="200">. We denote the discounted state visitation distribution for a policy _pi_ as <img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/rho_pi.png" width="30">
- Action Value Function (Q-function)  
  <img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/eq1.png" width="500">  
Recursive representation of Q function  
  <img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/eq2.png" width="500">  
If the policy is deterministic  
  <img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/eq3.png" width="500">  
The expectation depends only on the environment. This means that it is possible to learn Q offpolicy, using transitions which are generated from a different stochastic behavior policy.
- Loss for Q  
  <img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/eq4_5.png" width="500">
- Replay buffer
- Target network for y_t

## Algorithms
DDPG is an actor-critic based on the DPG algorithm.  
### DPG
The DPG algorithm maintains a parameterized actor function _mu(s)_ which specifies the current policy by deterministically mapping states to a specific action. The critic _Q(s,a)_ is learned using the Bellman equation as in Q-learning. The actor is updated by following the applying the chain rule to the expected return from the start distribution J with respect to the actor parameters.  
  <img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/eq6.png" width="500">  
The authors' contribution here is to provide modifications to DPG, inspired by the success of DQN, which allow it to use neural network function approximators to learn in large state and action spaces online.  

### Replay buffer
- In reinforcement learning, the assumption that the samples are independently and identically distributed does not hold, while most optimization algorithms assume it.
- Additionally, with mini-batches, we can make efficient use of hardware optimizations.
- To allow the algorithm to benefit from learning across a set of uncorrelated transitions, DDPG uses Replay buffer.
- Replay buffer stores transition tuple _(s\_t, a\_t, r\_t, s\_(t+1))_
- The buffer size can be large ~ 1e6.  


### Soft target updates
- Directly implementing Q learning with neural networks proved to be unstable in many environments. Q is prone to divergence.  
- The author's solution is __Soft target update__.
- They create a copy of the actor and critic networks that are used for calculating the target values. The weights of these target
networks are then updated by having them slowly track the learned networks.  
`p' = tau*p + (1-tau)*p' with tau << 1`  
- This means that the target values are constrained to change slowly, greatly improving the stability of learning.  

### Batch normalization
- When learning from low dimensional feature vector observations, the different components of the observation may have different physical units (for example, positions versus velocities) and the ranges may vary across environments.
- The solution is __Batch normalization__.
- It maintains a running average of the mean and variance to use for normalization during testing.

### Exploration - Ornstein-Uhlenbeck process
- A major challenge of learning in continuous action spaces is exploration.
- An advantage of offpolicies algorithms such as DDPG is that we can treat the problem of exploration independently from the learning algorithm.  
  <img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/eq7.png" width="200">  
- The authors used an Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein, 1930) to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia.

### DDPG algorithm and training process
<img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/algo1.png" width="800">   

## Results
- The algorithm is tested on simulations using MujoCo.  
<img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/fig1.png" width="800">  
  In order from the left: the cartpole swing-up task, a reaching task, a gasp and move task, a puck-hitting task, a monoped balancing task, two locomotion tasks and Torcs (driving simulator).  

- In all tasks, they ran experiments using both a low-dimensional state description (such as joint angles and positions) and high-dimensional renderings of the environment.  
  - They used action repeats as DQN for high dimensional renderings in order to make the problems approximately fully observable.  

- Below is the performance curve.  
<img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/fig2.png" width="800">  
  original DPG algorithm (minibatch NFQCA) with batch normalization (light grey), with target network (dark grey), with target networks and batch normalization (green), with target networks from pixel-only inputs (blue).    

- In particular, learning without a target network, as in the original work with DPG, is very poor in many environments.  

- Surprisingly, in some simpler tasks, learning policies from pixels is just as fast as learning using the low-dimensional state descriptor.  

- Below is performance table.  
<img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/table1.png" width="800">  
  - The performance is normalized using two baseline.  
  - The first baseline is the mean return from a naive policy which samples actions from a uniform distribution over the valid action space.  
  - The second baseline is iLQG (Todorov & Li, 2005), a planning based solver with full access to the underlying physical model and its derivatives.  
  
- It can be challenging to learn accurate value estimates. Q-learning, for example, is prone to overestimating values (Hasselt, 2010). This work is extended to TD3.  
<img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/fig3.png" width="800">   
  - In simple tasks DDPG estimates returns accurately without systematic biases. For harder tasks the Q estimates are worse, but DDPG is still able learn good policies.

## Conclusion
* __Contribution__ : The work combines insights from recent advances in deep learning and reinforcement learning, resulting in an algorithm that robustly solves challenging problems across a variety of domains with continuous action spaces.
* __Limitation__ : As with most model-free reinforcement approaches, DDPG requires a large number of training episodes to find solutions.

## Implementation Details
- Adam Optimizer with learning rate 10^-4 and 10^-3 for the actor and critic respectively.
- discount factor _gamma_ = 0.99
- soft target updates _tau_ = 0.001
- final output layer of the actor was a **tanh** layer
- 2 hidden layers with 400 and 300 units
- Actions were included at 2nd hidden layer of Q
- Final layer of actor and critic networks were initialized from a uniform distribution [-3x10^-3,3x10^-3]
- minibatch size 64
- replay buffer size 10^6
- Ornstein-Uhlenbeck process noise with _theta_=0.15 and _sigma_=0.2
- ~~Q : L_2 weight decay of 10^-2~~
- ~~Batch normalization~~  
For the stable training process, I didn't include l2 norm regularization and batch normalization in this implementation.

## Experiment Instruction
The agent is trained on OpenAI Gym Mujoco control tasks.
### Requirements
- Non-python
  - openCV
  - mujoco
- Python
  - gym
  - cv2
  - torch

The script for DDPG is at _ddpg_ directory. By typing the bash script below, you can train an agent for a specific task. The script will create a directory for the records of experiments(e.g. rewards, videos, graph)  
```
python ddpg.py --task=InvertedPendulum-v2
python ddpg.py --task=HalfCheetah-v2
```

## Experiment Results
You can find the trained agent below.
### Inverted Pendulum v2
<img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/ddpg/20190422_035056_InvertedPendulum-v2/rewards.png" width="600">   
<img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/inverted_pendulum.gif" width="300">   
https://youtu.be/KKp3w3aWsTA  


### Half Cheetah v2
<img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/ddpg/20190422_035344_HalfCheetah-v2/rewards.png" width="600">  
<img src="https://github.com/ehddnr747/duju-ml2-takehome/blob/master/images/half_cheetah.gif" width="300">   
https://youtu.be/iXSRwjpP3ak  
