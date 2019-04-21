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
DDPG is an actor-critic based on the DPG algorithm.  
### DPG
The DPG algorithm maintains a parameterized actor function _mu(s)_ which specifies the current policy by deterministically mapping states to a specific action. The critic _Q(s,a)_ is learned using the Bellman equation as in Q-learning. The actor is updated by following the applying the chain rule to the expected return from the start distribution J with respect to the actor parameters.  
![eq6]()  
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
networks are then updated by having them slowly track the learned networks. `p' = tau*p + (1-tau)*p' with tau << 1`.
- This means that the target values are constrained to change slowly, greatly improving the stability of learning.  

### Batch normalization
- When learning from low dimensional feature vector observations, the different components of the observation may have different physical units (for example, positions versus velocities) and the ranges may vary across environments.
- The solution is __Batch normalization__.
- It maintains a running average of the mean and variance to use for normalization during testing.

### Exploration - Ornstein-Uhlenbeck process
- A major challenge of learning in continuous action spaces is exploration.
- An advantage of offpolicies algorithms such as DDPG is that we can treat the problem of exploration independently from the learning algorithm.
![eq7]()  
- The authors used an Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein, 1930) to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia.

### DDPG algorithm and training process
![algo1]()  

## Results


## Related Work


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

## Experiment Results
