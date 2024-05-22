import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an ensemble of reward models
class EnsembleRewardModels:
    def __init__(self, num_models, input_dim, hidden_dim, output_dim):
        self.models = [RewardModel(input_dim, hidden_dim, output_dim) for _ in range(num_models)]
        self.optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in self.models]
        self.criterion = nn.MSELoss()
    
    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            for states, actions, rewards in data_loader:
                inputs = torch.cat((states, actions), dim=1)
                targets = rewards
                for model, optimizer in zip(self.models, self.optimizers):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
    
    def predict(self, state, action):
        inputs = torch.cat((state, action), dim=1)
        predictions = [model(inputs) for model in self.models]
        return predictions
    
    def compute_variance(self, state, action):
        predictions = self.predict(state, action)
        mean_prediction = torch.mean(torch.stack(predictions), dim=0)
        variance = torch.mean((torch.stack(predictions) - mean_prediction) ** 2, dim=0)
        return variance


