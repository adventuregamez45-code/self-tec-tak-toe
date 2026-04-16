import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Connect4Net(nn.Module):
    def __init__(self):
        super(Connect4Net, self).__init__()
        # Input: (batch, 1, 6, 7)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Flattened size: 128 * 6 * 7 = 5376
        self.fc1 = nn.Linear(5376, 512)
        
        self.action_head = nn.Linear(512, 7)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # x is (batch, 42), reshape to (batch, 1, 6, 7)
        x = x.view(-1, 1, 6, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        probs = F.softmax(self.action_head(x), dim=-1)
        value = torch.tanh(self.value_head(x))
        return probs, value

class Connect4Policy:
    def __init__(self):
        self.model = Connect4Net()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

    def select_action(self, state, valid_actions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self.model(state_tensor)
        
        mask = torch.zeros(7)
        mask[valid_actions] = 1.0
        
        masked_probs = probs.squeeze() * mask
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
        else:
            masked_probs = mask / mask.sum()

        m = torch.distributions.Categorical(masked_probs)
        action = m.sample().item()
        return action, m.log_prob(torch.tensor(action)), value.squeeze()
