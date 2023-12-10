import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import skimage.transform

import os
import sys

import itertools as it
import random
from collections import deque
from time import sleep, time
from tqdm import trange

import pickle

import vizdoom as vzd

from NETWORK import DuelQNet



DEVICE = "cpu"

def read_memory(MEMORY_FILE):
    loaded_mem = []
    with open(MEMORY_FILE, "rb") as fp:
        loaded_mem = pickle.load(fp)
    return loaded_mem

def write_memory(MEMORY_FILE, memory):
    with open(MEMORY_FILE, "wb") as fp:  
        pickle.dump(memory, fp)




########################################################################################################################################################################################################

class DQNAgent:
    def __init__(
        self,
        action_size,
        memory_size,
        batch_size,
        discount_factor,
        lr,
        load_model,
        model_savefile,
        epsilon,
        epsilon_decay,
        epsilon_min,
        MEMORY_FILE
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()
        self.model_savefile = model_savefile
        self.prev_memories = []

        study_list = read_memory(MEMORY_FILE)
        self.memory = deque(maxlen=memory_size+len(study_list))

        for e in study_list:
            self.memory.append(e)

        if load_model:
            print("Loading model from: ", self.model_savefile)
            self.q_net = torch.load(self.model_savefile)
            self.target_net = torch.load(self.model_savefile)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.q_net = DuelQNet(action_size).to(DEVICE)
            self.target_net = DuelQNet(action_size).to(DEVICE)

        self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_memory(self, filename):
        write_memory(filename, self.memory)

    def load_memory(self, filename):
        self.prev_memories.append( read_memory(filename) )
        
    def train(self): 
        self.train_memory(self.memory)
        for mem in self.prev_memories:
            self.train_memory(mem)

    def train_memory(self, memory_type):
        batch = random.sample(memory_type, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        row_idx = np.arange(self.batch_size)  # used for indexing the batch

        # value of the next states with double q learning
        # see https://arxiv.org/abs/1509.06461 for more information on double q learning
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        # this defines y = r + discount * max_a q(s', a)
        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

        # this selects only the q values of the actions taken
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)

        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

