import torch
import torch.nn as nn
import random
import numpy as np
from src.dqn.network import QNetwork
from src.dqn.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size, 
                 lr, gamma, epsilon, epsilon_decay, epsilon_min,
                 buffer_size, batch_size):
        
            self.q_network = QNetwork(input_size, hidden_size, output_size)
            self.target_network = QNetwork(input_size, hidden_size, output_size)
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
            self.buffer = ReplayBuffer(buffer_size)
            
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
            self.batch_size = batch_size

            self.n_actions = output_size

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = self.q_network.to(self.device)
            self.target_network = self.target_network.to(self.device)

    def select_action(self, state):
        random_num = random.random()  # no arguments needed, always returns 0-1
        if random_num <= self.epsilon:
            action = random.randint(0, self.n_actions-1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                action = self.q_network(state_tensor).argmax().item()
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        
    def train(self):
        if len(self.buffer) < self.batch_size:
            return None, None, None
        
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return loss.item(), current_q.mean().item(), grad_norm.item()
    

    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)