import torch
import numpy as np
import torch.nn.functional as F
from connect4_env import Connect4, find_critical_move
from connect4_policy import Connect4Policy
import random

def play_game(env, policy, epsilon=0.1, opponent_type="self"):
    state = env.reset()
    done = False
    history = []

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions: break
        
        current_player = env.current_player
        is_trainable = False
        
        # 1. Check for Critical Move (Win/Block)
        critical = find_critical_move(state, current_player)
        
        if opponent_type == "random" and current_player == -1:
            if critical is not None:
                action = critical
            else:
                action = random.choice(valid_actions)
        else:
            if critical is not None:
                action = critical
            elif random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                canonical_state = state * current_player
                action, _, _ = policy.select_action(canonical_state, valid_actions)
            is_trainable = True

        history.append({
            'state': (state * current_player).copy() if is_trainable else None,
            'action': action,
            'player': current_player,
            'valid_actions': valid_actions,
            'is_trainable': is_trainable
        })
        
        state, reward, done, info = env.step(action)

    winner = info.get('winner', 0)
    return history, winner

def train(policy, episodes=20000):
    epsilon = 0.5
    for eps in range(episodes):
        env = Connect4()
        opp_type = "random" if eps % 3 == 0 else "self"
        history, winner = play_game(env, policy, epsilon=epsilon, opponent_type=opp_type)
        epsilon = max(0.1, epsilon - 0.4 / (episodes * 0.8))
        
        if not history: continue

        policy.optimizer.zero_grad()
        total_loss = 0
        
        for i, entry in enumerate(history):
            if not entry['is_trainable']: continue

            if winner == 0:
                target_value = 0.0
            elif winner == entry['player']:
                target_value = 1.0
            else:
                target_value = -1.0
            
            # LOOK AHEAD: Failed block penalty
            if i + 1 < len(history):
                next_move = history[i+1]
                if winner != 0 and winner == next_move['player']:
                    target_value = -1.2

            state_tensor = torch.FloatTensor(entry['state']).unsqueeze(0)
            probs, value = policy.model(state_tensor)
            
            # Value Loss
            v_loss = F.mse_loss(value.squeeze(), torch.tensor(target_value, dtype=torch.float))
            
            # Policy Loss
            mask = torch.zeros(7)
            mask[entry['valid_actions']] = 1.0
            masked_probs = (probs.squeeze() * mask) + 1e-12
            masked_probs /= masked_probs.sum()
            
            log_prob = torch.log(masked_probs[entry['action']])
            advantage = target_value - value.detach().item()
            p_loss = -log_prob * advantage
            
            total_loss += (p_loss + v_loss)

        if total_loss != 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), max_norm=1.0)
            policy.optimizer.step()

        if (eps + 1) % 1000 == 0:
            print(f"Episode {eps+1}/{episodes} | Mode: {opp_type} | Loss: {total_loss.item():.4f} | Eps: {epsilon:.2f}")

if __name__ == "__main__":
    policy = Connect4Policy()
    # Resume training if possible, or start fresh
    try:
        policy.model.load_state_dict(torch.load("connect4_policy.pth"))
        print("Resuming training from connect4_policy.pth...")
    except:
        print("Starting fresh training...")

    print("Starting Advanced Connect 4 Training (20k episodes)...")
    train(policy, episodes=20000)
    torch.save(policy.model.state_dict(), "connect4_policy.pth")
    print("Training finished. Advanced Connect 4 Brain saved.")
