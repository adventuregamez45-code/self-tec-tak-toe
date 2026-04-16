import torch
import numpy as np
import torch.nn.functional as F
from environment import TicTacToe, find_critical_move
from policy import Policy
import random

def play_game(env, policy, epsilon=0.1, opponent_type="self"):
    state = env.reset()
    done = False
    history = [] 

    while not done:
        valid_indices = env.get_valid_actions()
        if not valid_indices: break
        current_player = env.current_player
        
        is_random_move = False
        if opponent_type == "random" and current_player == -1:
            # Random opponent still respects critical moves to keep games challenging
            critical = find_critical_move(state, current_player)
            if critical is not None:
                action_idx = critical
            else:
                action_idx = random.choice(valid_indices)
            is_random_move = True
        else:
            canonical_state = state * current_player
            
            # 1. CRITICAL MOVE DETECTION (YOUR FIX)
            critical = find_critical_move(state, current_player)
            
            if critical is not None:
                action_idx = critical
                # Even if critical, we still generate mask for training record
                mask = torch.zeros(9)
                mask[valid_indices] = 1.0
            else:
                state_tensor = torch.FloatTensor(canonical_state).unsqueeze(0)
                with torch.no_grad():
                    probs, _ = policy.model(state_tensor)
                
                mask = torch.zeros(9)
                mask[valid_indices] = 1.0
                masked_probs = probs.squeeze() * mask
                masked_probs[valid_indices] += 1e-12
                masked_probs /= masked_probs.sum()

                if random.random() < epsilon:
                    action_idx = random.choice(valid_indices)
                else:
                    m = torch.distributions.Categorical(masked_probs)
                    action_idx = m.sample().item()
            
            if action_idx not in valid_indices:
                action_idx = random.choice(valid_indices)

        if not is_random_move:
            history.append({
                'state': canonical_state.copy(),
                'action_idx': action_idx,
                'player': current_player,
                'mask': mask.clone(),
                'is_trainable': True
            })
        
        state, reward, done, info = env.step(action_idx)

    winner = info.get('winner', 0)
    return history, winner

def train(policy, episodes=50000):
    optimizer = policy.optimizer
    epsilon = 0.5
    
    for eps in range(episodes):
        env = TicTacToe()
        opp_type = "random" if eps % 3 == 0 else "self"
        history, winner = play_game(env, policy, epsilon=epsilon, opponent_type=opp_type)
        epsilon = max(0.05, epsilon - 0.45 / (episodes * 0.8))
        
        if not history: continue

        optimizer.zero_grad()
        total_loss = 0
        
        for i, entry in enumerate(history):
            if winner == 0:
                target_value = 0.0
            elif winner == entry['player']:
                target_value = 1.0
            else:
                target_value = -1.0
            
            # Failed block look-ahead penalty (User's fix)
            if i + 1 < len(history):
                next_move = history[i+1]
                if winner != 0 and winner == next_move['player']:
                    target_value = -1.2 
            
            state_tensor = torch.FloatTensor(entry['state']).unsqueeze(0)
            probs, value = policy.model(state_tensor)
            
            v_loss = F.mse_loss(value.squeeze(), torch.tensor(target_value, dtype=torch.float))
            
            mask = entry['mask']
            masked_probs = probs.squeeze() * mask
            valid_indices = torch.where(mask > 0)[0]
            masked_probs[valid_indices] += 1e-12
            masked_probs /= masked_probs.sum()
            
            log_prob = torch.log(masked_probs[entry['action_idx']])
            advantage = target_value - value.detach().item()
            p_loss = -log_prob * advantage
            
            total_loss += (p_loss + v_loss)

        if total_loss != 0 and not torch.isnan(total_loss):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), max_norm=1.0)
            optimizer.step()

        if (eps + 1) % 5000 == 0:
            print(f"Episode {eps+1}/{episodes} | Mode: {opp_type} | Loss: {total_loss.item():.4f} | Eps: {epsilon:.2f}")

if __name__ == "__main__":
    policy = Policy(hidden_size=2000)
    policy.optimizer = torch.optim.Adam(policy.model.parameters(), lr=0.0001)
    
    print("Starting Rule-Augmented Defensive Training...")
    train(policy, episodes=50000)
    
    torch.save(policy.model.state_dict(), "tictactoe_policy.pth")
    print("Training finished. Hybrid brain saved.")
