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
        
            # Twee netwerken: een om acties te kiezen, een als stabiel doel
            self.q_network = QNetwork(input_size, hidden_size, output_size)
            self.target_network = QNetwork(input_size, hidden_size, output_size)
            
            # Adam optimizer voor de gewichten
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
            
            # Replay buffer
            self.buffer = ReplayBuffer(buffer_size)
            
            # Hyperparameters opslaan
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
            self.batch_size = batch_size

            # Aantal acties
            self.n_actions = output_size

            # GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = self.q_network.to(self.device)
            self.target_network = self.target_network.to(self.device)

    def select_action(self, state):
        random_num = random.random()
        
        # Epsilon-greedy: 
        if random_num <= self.epsilon:
            # Explore
            action = random.randint(0, self.n_actions-1)
        else:
            # Exploit
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                action = self.q_network(state_tensor).argmax().item()
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        # Save
        self.buffer.push(state, action, reward, next_state, done)
        
    def train(self):
        # Wacht totdat de buffer genoeg ervaringen heeft
        if len(self.buffer) < self.batch_size:
            return None, None, None
        
        # Sample een willekeurige batch uit de replay buffer
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Zet om naar tensors op het juiste apparaat (GPU/CPU)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Bereken de huidige Q-waarden voor de genomen acties
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double-DQN
        with torch.no_grad():
            # Stap 1: Beste actie
            next_actions = self.q_network(next_states).argmax(1)
            # Stap 2: Target netwerk
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Bellman vergelijking
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q.unsqueeze(1))
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return loss.item(), current_q.mean().item(), grad_norm.item()
    

    def save(self, path):
        # Sla het model op zodat we later verder kunnen trainen
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        # Laad een eerder opgeslagen model om verder te trainen
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

    def update_target_network(self):
        # Kopieer de gewichten van het Q-netwerk naar het target netwerk
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        # Verlaag epsilon na elke episode, maar nooit onder het minimum
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)