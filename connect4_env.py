import numpy as np

class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # 1 for Red, -1 for Yellow
        self.winner = None

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.winner = None
        return self.board.copy().flatten()

    def get_valid_actions(self):
        # A column is valid if the top row is empty
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def step(self, col):
        if self.board[0, col] != 0:
            raise ValueError(f"Column {col} is full!")

        # Find the lowest empty row in this column
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = self.current_player
                row = r
                break
        
        if self.check_winner(row, col):
            self.winner = self.current_player
            return self.board.copy().flatten(), 1, True, {"winner": self.winner}
        elif len(self.get_valid_actions()) == 0:
            self.winner = 0
            return self.board.copy().flatten(), 0, True, {"winner": 0}
        else:
            self.current_player *= -1
            return self.board.copy().flatten(), 0, False, {}

    def check_winner(self, r, c):
        p = self.current_player
        # Directions: (dr, dc)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # Check one way
            for i in range(1, 4):
                nr, nc = r + dr*i, c + dc*i
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr, nc] == p:
                    count += 1
                else:
                    break
            # Check the other way
            for i in range(1, 4):
                nr, nc = r - dr*i, c - dc*i
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr, nc] == p:
                    count += 1
                else:
                    break
            if count >= 4:
                return True
        return False

    def render(self):
        symbols = {1: 'R', -1: 'Y', 0: '.'}
        print("\n 0 1 2 3 4 5 6")
        for r in range(self.rows):
            print("|" + "|".join(symbols[val] for val in self.board[r]) + "|")
        print("-" * 15)

def find_critical_move(board_flat, player):
    """
    Finds a move that wins immediately or blocks an immediate win.
    board_flat: 42-element array
    """
    board = board_flat.reshape(6, 7)
    opponent = -player
    
    # Priority 1: Win
    move = _check_threats(board, player)
    if move is not None: return move
    
    # Priority 2: Block
    move = _check_threats(board, opponent)
    if move is not None: return move
    
    return None

def _check_threats(board, p):
    """Checks if player p can win in the next move."""
    rows, cols = 6, 7
    # For each column, find where the piece would land
    for c in range(cols):
        r = -1
        for row_idx in range(rows-1, -1, -1):
            if board[row_idx, c] == 0:
                r = row_idx
                break
        
        if r == -1: continue # Column full
        
        # Check if placing p at (r, c) creates a 4-in-a-row
        if _would_win(board, r, c, p):
            return c
    return None

def _would_win(board, r, c, p):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        for i in range(1, 4):
            nr, nc = r + dr*i, c + dc*i
            if 0 <= nr < 6 and 0 <= nc < 7 and board[nr, nc] == p:
                count += 1
            else:
                break
        for i in range(1, 4):
            nr, nc = r - dr*i, c - dc*i
            if 0 <= nr < 6 and 0 <= nc < 7 and board[nr, nc] == p:
                count += 1
            else:
                break
        if count >= 4:
            return True
    return False

if __name__ == "__main__":
    env = Connect4()
    env.step(3)
    env.step(3)
    env.step(2)
    env.render()
