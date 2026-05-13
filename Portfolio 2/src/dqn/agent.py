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
        # Don't train if buffer doesn't have enough samples yet
        if len(self.buffer) < self.batch_size:
            return
        
        # 1. Sample a batch from the replay buffer
        batch = self.buffer.sample(self.batch_size)
        
        # 2. Unpack the batch into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 3. Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 4. Compute current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 5. Compute target Q-values using target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 6. Compute loss
        loss = nn.MSELoss()(current_q, target_q.unsqueeze(1))
        
        # 7. Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)