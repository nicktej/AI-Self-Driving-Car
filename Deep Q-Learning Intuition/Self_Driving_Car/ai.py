# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) #input to hidden1
        self.fc2 = nn.Linear(30, 60) #hidden1 to hidden2
        self.fc3 = nn.Linear(60, 120) #hidden2 to hidden3
        self.fc4 = nn.Linear(120, nb_action) #hidden3 to output
    
    def forward(self, state):
        x = F.relu(self.fc1(state)) #rectified linear unit
        y = F.relu(self.fc2(x))
        z = F.relu(self.fc3(y))
        q_values = self.fc4(z)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] 
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma): #initializing the Q-learning variables
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*75) # T=100
        # T for temperature. T = 0 means no AI. The higher the T, the more accurate the path, but the less 
        # the AI will explore
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) #first part of Q-learning formula
        next_outputs = self.model(batch_next_state).detach().max(1)[0] #second part of Q-learning
        target = self.gamma*next_outputs + batch_reward #calculation
        td_loss = F.smooth_l1_loss(outputs, target) #temporal difference
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True) #backpropagation
        self.optimizer.step() #updates the weights
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action #updating action
        self.last_state = new_state #updating state
        self.last_reward = reward #updating reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0] #adding to the reward list
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")