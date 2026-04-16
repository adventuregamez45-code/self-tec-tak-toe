import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        self.winner = None

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.winner = None
        return self.board.copy()

    def get_state(self):
        return self.board.copy()

    def get_valid_actions(self):
        return np.where(self.board == 0)[0].tolist()

    def step(self, action_idx):
        if self.board[action_idx] != 0:
            raise ValueError(f"Invalid action {action_idx}: position already taken.")
        self.board[action_idx] = self.current_player
        if self.check_winner():
            self.winner = self.current_player
            return self.get_state(), 1, True, {"winner": self.winner}
        elif np.all(self.board != 0):
            self.winner = 0
            return self.get_state(), 0, True, {"winner": 0}
        else:
            self.current_player *= -1
            return self.get_state(), 0, False, {}

    def check_winner(self):
        b = self.board.reshape(3, 3)
        p = self.current_player
        for i in range(3):
            if np.all(b[i, :] == p) or np.all(b[:, i] == p): return True
        if b[0,0] == b[1,1] == b[2,2] == p: return True
        if b[0,2] == b[1,1] == b[2,0] == p: return True
        return False

    def render(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        b = self.board
        print("-" * 13)
        for i in range(0, 9, 3):
            print(f"| {symbols[b[i]]} | {symbols[b[i+1]]} | {symbols[b[i+2]]} |")
            print("-" * 13)

def count_threats(board, player):
    """Count how many lines have exactly 2 of player's pieces and 1 empty."""
    lines = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    threats = 0
    for line in lines:
        vals = [board[i] for i in line]
        if vals.count(player) == 2 and vals.count(0) == 1:
            threats += 1
    return threats

def find_fork(board, player):
    empty_cells = [i for i in range(9) if board[i] == 0]
    fork_moves = []
    for move in empty_cells:
        test_board = board.copy()
        test_board[move] = player
        if count_threats(test_board, player) >= 2:
            fork_moves.append(move)
    return fork_moves

def find_critical_move(board, player):
    opponent = -player
    lines = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    
    # Priority 1: Win immediately
    for line in lines:
        vals = [board[i] for i in line]
        if vals.count(player) == 2 and vals.count(0) == 1:
            return line[vals.index(0)]
    
    # Priority 2: Block opponent's immediate win
    for line in lines:
        vals = [board[i] for i in line]
        if vals.count(opponent) == 2 and vals.count(0) == 1:
            return line[vals.index(0)]
    
    # Priority 3: Play a fork for yourself
    my_forks = find_fork(board, player)
    if my_forks:
        return my_forks[0]
    
    # Priority 4: Block opponent's fork
    opp_forks = find_fork(board, opponent)
    if opp_forks:
        # Complex block: Force them to respond to our threat instead of making a fork
        if len(opp_forks) > 1:
            for line in lines:
                vals = [board[i] for i in line]
                if vals.count(player) == 1 and vals.count(0) == 2:
                    empty_in_line = [cell for cell in line if board[cell] == 0]
                    for cell in empty_in_line:
                        if cell not in opp_forks:
                            return cell
        return opp_forks[0]
    
    return None
