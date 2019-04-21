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
  - return(discounted future reward) ![return]()
  - discounting factor _gamma_ in [0,1]
- Goal is maximizing the expected return from the start distribution ![J](). We denote the discounted state visitation distribution for a policy _pi_ as ![rho^pi]()
- Action Value Function (Q-function) ![eq1]()  
Recursive representation of Q function ![eq2]()  
If the policy is deterministic ![eq3]()  
The expectation depends only on the environment. This means that it is possible to learn Q offpolicy, using transitions which are generated from a different stochastic behavior policy.
- Loss for Q
![eq4]()  
![eq5]()
- Replay buffer
- Target network for y_t

## Algorithms
- The algorithms is based on DPG. DPG maintais a para

## Results


## Related Work


## Conclusion


## Implementation Details
- Adam Optimizer with learning rate 10^-4 and 10^-3 for the actor and critic respectively.
- Q : L_2 weight decay of 10^-2
- discount factor _gamma_ = 0.99
- soft target updates _tau_ = 0.001
- final output layer of the actor was a **tanh** layer
- 2 hidden layers with 400 and 300 units
- Actions were included at 2nd hidden layer of Q
- Final layer of actor and critic networks were initialized from a uniform distribution [-3x10^-3,3x10^-3]
- minibatch size 64
- replay buffer size 10^6
- Ornstein-Uhlenbeck process noise with _theta_=0.15 and _sigma_=0.2
