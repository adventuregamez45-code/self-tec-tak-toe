import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, hidden_size=2000):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(9, hidden_size)
        
        # Policy head: 9 move probabilities
        self.action_head = nn.Linear(hidden_size, 9)
        
        # Value head: scalar predicting win/loss/draw probability (-1 to 1)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        probs = F.softmax(self.action_head(x), dim=-1)
        value = torch.tanh(self.value_head(x)) # Bound value between -1 and 1
        
        return probs, value

class Policy:
    def __init__(self, hidden_size=2000):
        self.model = PolicyNet(hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def select_action(self, state, valid_actions):
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        probs, value = self.model(state_tensor)
        
        mask = torch.zeros(9)
        valid_indices = [r * 3 + c for r, c in valid_actions]
        mask[valid_indices] = 1.0
        
        masked_probs = probs.squeeze(0) * mask
        if masked_probs.sum() == 0:
            masked_probs = mask / mask.sum()
        else:
            masked_probs = masked_probs / masked_probs.sum()

        m = torch.distributions.Categorical(masked_probs)
        action_idx = m.sample()
        
        row, col = divmod(action_idx.item(), 3)
        return (row, col), m.log_prob(action_idx), value.squeeze(0)

if __name__ == "__main__":
    policy = Policy()
    dummy_state = np.zeros((3, 3))
    valid_moves = [(0, 0), (1, 1), (2, 2)]
    
    action, log_prob, value = policy.select_action(dummy_state, valid_moves)
    print(f"Selected action: {action}")
    print(f"Predicted State Value: {value.item():.4f}")
