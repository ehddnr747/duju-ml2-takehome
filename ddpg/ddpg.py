import gym
import os
import matplotlib.pyplot as plt

from collections import deque
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2
import datetime
import csv
import argparse


class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr, device):
        super(DDPGActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.device = device

        self.fc1 = nn.Linear(self.state_dim, 400).to(device)
        self.fc2 = nn.Linear(400, 300).to(device)
        self.fc3 = nn.Linear(300, self.action_dim).to(device)

        # self.bn_input = nn.BatchNorm1d(self.state_dim).to(device)
        # self.bn1 = nn.BatchNorm1d(400).to(device)
        # self.bn2 = nn.BatchNorm1d(300).to(device)

        nn.init.uniform_(tensor=self.fc3.weight, a=-3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.fc3.bias, a=-3e-3, b=3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.actor_lr)

    def forward(self, x):
        # x = self.bn_input(x)
        x = F.relu(self.fc1(x))
        # x = self.bn1(x)
        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        x = torch.tanh(self.fc3(x))

        return x


class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, critic_lr, device):
        super(DDPGCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.critic_lr = critic_lr
        self.device = device

        self.fc1 = nn.Linear(state_dim, 400).to(device)
        self.fc2 = nn.Linear(400 + self.action_dim, 300).to(device)
        self.fc3 = nn.Linear(300, 1).to(device)

        # self.bn_input = nn.BatchNorm1d(state_dim).to(device)
        # self.bn1 = nn.BatchNorm1d(400).to(device)

        nn.init.uniform_(tensor=self.fc3.weight, a=-3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.fc3.bias, a=-3e-3, b=3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.critic_lr)

    def forward(self, x, a):
        # x = self.bn_input(x)
        x = F.relu(self.fc1(x))
        # x = self.bn1(x)
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def soft_target_update(main, target, tau):
    params_main = list(main.parameters())
    params_target = list(target.parameters())

    assert len(params_main) == len(params_target)

    for pi in range(len(params_main)):
        params_target[pi].data.copy_((1 - tau) * params_target[pi].data + tau * params_main[pi].data)


def target_initialize(main, target):
    params_main = list(main.parameters())
    params_target = list(target.parameters())

    assert len(params_main) == len(params_target)

    for pi in range(len(params_main)):
        params_target[pi].data.copy_(params_main[pi].data)


def bn_stat_sync(main, target):
    target.bn_input.running_mean = main.bn_input.running_mean
    target.bn_input.running_var = main.bn_input.running_var

    target.bn1.running_mean = main.bn1.running_mean
    target.bn1.running_var = main.bn1.running_var

    if hasattr(target, 'bn2'):
        target.bn2.running_mean = main.bn2.running_mean
        target.bn2.running_var = main.bn2.running_var


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None, actions_per_control=1):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
        self.actions_per_control = actions_per_control

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def __call__(self):
        if self.actions_per_control == 1:  # return 1d array
            return self.sample()
        else:  # return 2d array
            noises = []
            for _ in range(self.actions_per_control):
                noises.append(self.sample())
            return np.array(noises)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):

        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class VideoSaver:
    def __init__(self, save_path, source_fps, target_fps=30, width=None, height=None):

        self.save_path = save_path
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.counter = 0

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.save_path, self.fourcc, self.target_fps, (self.width, self.height))

        self.save_frame_index = uniform_fps_downsizer(self.source_fps, self.target_fps)

    def write(self, frame):
        if (self.counter) % self.source_fps in self.save_frame_index:
            self.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            pass
        self.counter += 1

    def release(self):
        self.out.release()


def RGB2BGR(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def uniform_fps_downsizer(source_fps, target_fps):
    assert type(source_fps) == int
    interval = source_fps / float(target_fps)
    index_list = []

    for i in range(target_fps):
        index_list.append(int(np.round(i * interval)))

    return index_list


def append_file_writer(dirpath, file_name, _str):
    print(_str, end="")
    with open(os.path.join(dirpath, file_name), "a") as f:
        f.write(_str)


def experiment_name(domain_name):
    dt = datetime.datetime.now()
    exp_name = str(dt.year) + str(dt.month).zfill(2) + str(dt.day).zfill(2) + \
               "_" + str(dt.hour).zfill(2) + str(dt.minute).zfill(2) + str(dt.second).zfill(2) + "_" + domain_name

    return exp_name


def save_graph(record_dir):
    target_dir = record_dir

    assert os.path.isdir(target_dir)

    txt_path = os.path.join(target_dir, "rewards.txt")
    image_path = os.path.join(target_dir, "rewards.png")

    with open(txt_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter='*')
        rewards = []
        for row in csvreader:
            rewards.append(float(row[3]))

    with open(txt_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter='*')
        max_q_values = []
        for row in csvreader:
            max_q_values.append(float(row[6]))

    with open(txt_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter='*')
        eval_rewards = []
        for row in csvreader:
            eval_rewards.append(float(row[9]))

    if np.max(rewards) <= 1000:
        reward_scale = 1000
    else:
        reward_scale = 10000

    draw(rewards, max_q_values, eval_rewards, reward_scale)

    plt.savefig(image_path, dpi=300)
    plt.close()


def draw(rewards, max_q_values, eval_rewards, reward_scale):
    t = np.arange(1, len(rewards) + 1)

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('episode')
    ax1.set_ylabel('reward', color=color)
    ax1.plot(t, rewards, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, reward_scale])

    color = 'tab:orange'
    ax1.plot(t, eval_rewards, color=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('max_q', color=color)
    ax2.plot(t, max_q_values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, np.max([np.max(max_q_values), 1.0])])

    fig.tight_layout()


def train(actor_main, critic_main, actor_target, critic_target, replay_buffer, batch_size):
    actor_main.train()
    critic_main.train()
    actor_target.eval()
    critic_target.eval()

    # bn_stat_sync(main = actor_main, target = actor_target)
    # bn_stat_sync(main = critic_main, target = critic_target)

    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batch_size)

    s_batch = torch.FloatTensor(s_batch).to(device)
    a_batch = torch.FloatTensor(a_batch).to(device)
    r_batch = torch.FloatTensor(r_batch).to(device)
    s2_batch = torch.FloatTensor(s2_batch).to(device)

    with torch.no_grad():
        y_i = torch.FloatTensor(np.zeros([batch_size, 1])).to(device)

        next_target_q = critic_target.forward(s2_batch,
                                              actor_target.forward(s2_batch)
                                              )

        for batch_iter in range(batch_size):
            if t_batch[batch_iter]:
                y_i[batch_iter] = r_batch[batch_iter]
            else:
                y_i[batch_iter] = r_batch[batch_iter] + 0.99 * next_target_q[batch_iter]

    q = critic_main.forward(s_batch, a_batch)

    criterion = nn.MSELoss()

    loss = criterion(q, y_i)

    critic_main.optimizer.zero_grad()
    loss.backward()
    critic_main.optimizer.step()

    actor_main.optimizer.zero_grad()
    a_out = actor_main.forward(s_batch)
    loss = -critic_main.forward(s_batch, a_out).mean()

    loss.backward()
    actor_main.optimizer.step()

    soft_target_update(actor_main, actor_target, 0.001)
    soft_target_update(critic_main, critic_target, 0.001)

    return np.max(q.detach().cpu().numpy())


def evaluation(actor_main, env, action_scale, video_flag=False):
    s = env.reset()
    epi_reward = 0.0

    s = torch.FloatTensor(s).to(device)

    if video_flag:
        frame = env.render(mode="rgb_array")
        height = frame.shape[0]
        width = frame.shape[1]
        video_saver = VideoSaver(
            os.path.join(dir_path, "video_" + str(epi_count) + ".avi"),
            100, target_fps=30, width=width, height=height
        )
        video_saver.write(frame)

    while (True):
        with torch.no_grad():
            a = actor_main.forward(s.view([1, -1])).cpu().numpy()
            a = a[0] * action_scale

        s2, r, done, _ = env.step(a)
        if video_flag:
            video_saver.write(env.render(mode="rgb_array"))

        s = torch.FloatTensor(s2).to(device)

        epi_reward += r

        if done:
            if video_flag:
                video_saver.release()
            break

    return epi_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="arg for task")

    parser.add_argument('--task', help='which task?', default="InvertedPendulum-v2")

    task_name = parser.parse_args().task
    batch_size = 64
    device = torch.device("cuda")

    exp_name = experiment_name(task_name)
    print("Start", exp_name)

    env = gym.make(task_name)
    replay_buffer = ReplayBuffer(buffer_size=1000000)

    assert len(env.observation_space.shape) == 1
    assert len(env.action_space.shape) == 1
    assert (np.abs(env.action_space.high) == np.abs(env.action_space.low)).all()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = env.action_space.high

    actor_lr = 1e-4
    critic_lr = 1e-3

    actor_main = DDPGActor(state_dim, action_dim, actor_lr, device)
    actor_target = DDPGActor(state_dim, action_dim, actor_lr, device)
    critic_main = DDPGCritic(state_dim, action_dim, critic_lr, device)
    critic_target = DDPGCritic(state_dim, action_dim, critic_lr, device)

    target_initialize(actor_main, actor_target)
    target_initialize(critic_main, critic_target)

    iter_count = 0
    epi_count = 0
    max_q = 0

    noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros([action_dim]), sigma=0.2, theta=.15)

    if not os.path.isdir(os.path.join(".", exp_name)):
        os.mkdir(os.path.join(".", exp_name))

    dir_path = os.path.join(".", exp_name)

    while (iter_count < 1000000):

        noise.reset()
        s = env.reset()
        epi_reward = 0.0

        s = torch.FloatTensor(s).to(device)

        # inside episode
        while (True):

            actor_main.eval()

            with torch.no_grad():
                a = actor_main.forward(s.view(1, -1)).cpu().numpy().reshape([action_dim])
                a = a + noise()
                a = a * action_scale
                a = np.clip(a, -action_scale, action_scale)

            s2, r, done, _ = env.step(a)
            # [observation, reward, done, info]

            replay_buffer.add(s.cpu().numpy(), a, r, done, s2)

            s = torch.FloatTensor(s2).to(device)
            epi_reward += r

            iter_count += 1

            if replay_buffer.size() >= batch_size:
                max_q = train(actor_main, critic_main, actor_target, critic_target, replay_buffer, batch_size)

            if done:
                break

        if epi_count % 50 == 0:
            video_flag = True
        else:
            video_flag = False

        eval_return = evaluation(actor_main, env, action_scale, video_flag)

        append_file_writer(dir_path, "rewards.txt",
                           str(epi_count) + " *** " + str(epi_reward) + " *** " \
                           + str(max_q) + " *** " + str(eval_return) + "\n")

        if epi_count % 50 == 0:
            save_graph(dir_path)

        epi_count += 1