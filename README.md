# duju-ml2-takehome
Donguk Ju's take home project for ml2 internship


# Review
# CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING

## Summary
In the paper the authors propsoe an Actor-Critic, off-policy model-free algorithms based on DPG[(Deterministic Policy Gradient)](http://proceedings.mlr.press/v32/silver14.pdf) for continuous action space inspired by Deep Q-Learning[(DQN)](https://arxiv.org/abs/1312.5602).

## Introduction
- While DQN solves problems with high-dimensional observation spaces, it can only handle discrete and low-dimensional action spaces. Many tasks of interest, most notably physical control tasks, have **continuous** (real valued) and high dimensional action spaces.
- An obvious approach to adapting deep reinforcement learning methods such as DQN to continuous domains is to to simply discretize the action space. However, this has many limitations, most notably **the curse of dimensionality**: the number of actions increases exponentially with the number of degrees of freedom.
- In this work the authors present a **model-free, off-policy actor-critic algorithm** using deep function approximators that can learn policies in high-dimensional, continuous action spaces.
- A naive application of this actor-critic method with neural function approximators is **unstable** for challenging problems. Prior to DQN, it was generally believed that learning value functions using large, non-linear function approximators was difficult and unstable.
- Two innovation in this method
  1. The network is trained **off-policy** with samples from a replay buffer to **minimize correlations between samples**.
  2. The network is trained with a **target Q network** to give consistent targets during temporal difference backups.
- Proposed model-free approach which we call Deep DPG (DDPG) can learn **competitive policies for all of our tasks using low-dimensional observations (e.g. cartesian coordinates or joint angles) using the same hyper-parameters and network structure.**
