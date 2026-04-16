import torch
import numpy as np
import random
from connect4_env import Connect4, find_critical_move
from connect4_policy import Connect4Policy

def get_human_action(valid_actions):
    while True:
        try:
            choice = input(f"Choose a column {valid_actions}: ")
            col = int(choice)
            if col in valid_actions:
                return col
            else:
                print("Invalid column or column full.")
        except ValueError:
            print("Please enter a single number.")

def get_difficulty():
    print("\nSelect Connect 4 Difficulty:")
    print("1: Easy (Random)")
    print("2: Medium (CNN Only - Pure Pattern Recognition)")
    print("3: Hard (Rules + CNN - High Defensive Priority)")
    while True:
        choice = input("Choice (1-3): ")
        if choice in ['1', '2', '3']:
            return int(choice)
        print("Invalid choice.")

def play():
    policy = Connect4Policy()
    try:
        policy.model.load_state_dict(torch.load("connect4_policy.pth"))
        print("Connect 4 Brain loaded.")
    except FileNotFoundError:
        print("No brain found. Play on Easy or train first.")
        # We can still play Easy
    
    policy.model.eval()
    difficulty = get_difficulty()
    game_count = 1

    while True:
        env = Connect4()
        state = env.reset() # 42-element flat array
        done = False
        
        ai_starts = (game_count % 2 != 0)
        ai_player_id = 1 if ai_starts else -1
        human_player_id = -ai_player_id
        
        print(f"\n--- Connect 4 Game {game_count} | Difficulty: {['','Easy','Medium','Hard'][difficulty]} ---")
        if ai_starts: print("AI starts as 'R' (Red).")
        else: print("You start as 'R' (Red).")
        
        env.render()

        while not done:
            current_player = env.current_player
            valid_actions = env.get_valid_actions()

            if current_player == ai_player_id:
                print("AI is thinking...")
                action = None
                
                if difficulty == 1:
                    action = random.choice(valid_actions)
                elif difficulty == 2:
                    canonical_state = state * current_player
                    action, _, _ = policy.select_action(canonical_state, valid_actions)
                elif difficulty == 3:
                    critical = find_critical_move(state, current_player)
                    if critical is not None:
                        action = critical
                        print(f"AI plays CRITICAL move in column {action}")
                    else:
                        canonical_state = state * current_player
                        action, _, _ = policy.select_action(canonical_state, valid_actions)
                
                print(f"AI moves to column {action}")
            else:
                action = get_human_action(valid_actions)

            state, reward, done, info = env.step(action)
            env.render()

        winner = info.get('winner', 0)
        if winner == ai_player_id: print("Outcome: AI Wins!")
        elif winner == human_player_id: print("Outcome: You Won!")
        else: print("Outcome: It's a Draw!")

        cont = input("\nPlay again? (y/n): ").lower()
        if cont != 'y': break
        
        change_diff = input("Change difficulty? (y/n): ").lower()
        if change_diff == 'y': difficulty = get_difficulty()
        game_count += 1

if __name__ == "__main__":
    play()
