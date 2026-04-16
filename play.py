import torch
import numpy as np
import random
from environment import TicTacToe, find_critical_move
from policy import Policy

def get_human_action(valid_indices):
    while True:
        try:
            move = input("Enter your move (row and col, e.g., '0 1'): ")
            row, col = map(int, move.split())
            idx = row * 3 + col
            if idx in valid_indices:
                return idx
            else:
                print("Invalid move! That spot is taken or out of bounds.")
        except ValueError:
            print("Please enter two numbers separated by a space (0, 1, or 2).")

def get_difficulty():
    print("\nSelect Difficulty:")
    print("1: Easy (Random)")
    print("2: Medium (Neural Net Only - Pure Intuition)")
    print("3: Hard (Rules + Neural Net - Unbeatable)")
    while True:
        choice = input("Choice (1-3): ")
        if choice in ['1', '2', '3']:
            return int(choice)
        print("Invalid choice. Enter 1, 2, or 3.")

def play():
    policy = Policy(hidden_size=2000)
    try:
        policy.model.load_state_dict(torch.load("tictactoe_policy.pth"))
        print("System loaded successfully.")
    except FileNotFoundError:
        print("Could not find 'tictactoe_policy.pth'. Train first.")
        return

    policy.model.eval()
    difficulty = get_difficulty()
    game_count = 1

    while True:
        env = TicTacToe()
        state = env.reset()
        done = False
        
        ai_starts = (game_count % 2 != 0)
        ai_player_id = 1 if ai_starts else -1
        human_player_id = -ai_player_id
        
        print(f"\n--- Game {game_count} | Difficulty: {['','Easy','Medium','Hard'][difficulty]} ---")
        if ai_starts: print("AI starts as 'X'.")
        else: print("You start as 'X'.")
        
        env.render()

        while not done:
            current_player = env.current_player
            valid_indices = env.get_valid_actions()

            if current_player == ai_player_id:
                print("AI is thinking...")
                
                action_idx = None
                
                # DIFFICULTY LOGIC
                if difficulty == 1: # EASY: Random
                    action_idx = random.choice(valid_indices)
                    print(f"AI (Easy) moves to ({action_idx//3}, {action_idx%3})")
                
                elif difficulty == 2: # MEDIUM: Neural Net Only
                    canonical_state = state * current_player
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(canonical_state).unsqueeze(0)
                        probs, _ = policy.model(state_tensor)
                        mask = torch.zeros(9)
                        mask[valid_indices] = 1.0
                        masked_probs = probs.squeeze() * mask
                        action_idx = torch.argmax(masked_probs).item()
                    print(f"AI (Medium) moves to ({action_idx//3}, {action_idx%3})")
                
                elif difficulty == 3: # HARD: Rules + Net
                    critical = find_critical_move(state, current_player)
                    if critical is not None:
                        action_idx = critical
                        print(f"AI (Hard) plays CRITICAL move at ({action_idx//3}, {action_idx%3})")
                    else:
                        canonical_state = state * current_player
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(canonical_state).unsqueeze(0)
                            probs, _ = policy.model(state_tensor)
                            mask = torch.zeros(9)
                            mask[valid_indices] = 1.0
                            masked_probs = probs.squeeze() * mask
                            action_idx = torch.argmax(masked_probs).item()
                        print(f"AI (Hard) plays POSITIONAL move at ({action_idx//3}, {action_idx%3})")
            else:
                action_idx = get_human_action(valid_indices)

            state, reward, done, info = env.step(action_idx)
            env.render()

        winner = info.get('winner', 0)
        if winner == ai_player_id: print("Outcome: AI Wins!")
        elif winner == human_player_id: print("Outcome: You Won!")
        else: print("Outcome: It's a Draw!")

        cont = input("\nPlay again? (y/n): ").lower()
        if cont != 'y': break
        
        # Optionally allow changing difficulty between games
        change_diff = input("Change difficulty? (y/n): ").lower()
        if change_diff == 'y':
            difficulty = get_difficulty()
            
        game_count += 1
        print("\n" + "="*20 + "\n")

if __name__ == "__main__":
    play()
