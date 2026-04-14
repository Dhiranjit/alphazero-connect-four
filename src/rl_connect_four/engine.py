"""
Pure game engine for Connect Four.

Defines the board representation and all game mechanics: placing pieces,
checking for wins, detecting draws, and listing legal moves. No learning,
no agents — just the rules of the game.

All functions are stateless and return new boards rather than mutating,
so they are safe to use from any RL algorithm.
"""

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2
WIN_LENGTH = 4


def create_board():
    return [[EMPTY] * COLS for _ in range(ROWS)]


def drop_piece(board, col, player):
    # Scan from the bottom up — pieces settle at the lowest empty row
    for row in range(ROWS - 1, -1, -1):
        if board[row][col] == EMPTY:
            new_board = [r[:] for r in board]
            new_board[row][col] = player
            return new_board


def get_legal_actions(board):
    return [col for col in range(COLS)
            if board[0][col] == EMPTY]


def check_win(board, player):
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3): # To prevent out of bound error
            if all(board[r][c + i] == player for i in range(WIN_LENGTH)):
                return True
            
    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r + i][c] == player for i in range(WIN_LENGTH)):
                return True
    
    # Diagonal: top-left to bottom-right
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == player for i in range(WIN_LENGTH)):
                return True
    
    # Diagonal: bottom-left to top-rigth
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i][c + i] == player for i in range(WIN_LENGTH)):
                return True
            
    return False


def is_draw(board):
    # No legal moves left and no winner — board is full
    return len(get_legal_actions(board)) == 0
    

def other_player(player):
    return PLAYER2 if player == PLAYER1 else PLAYER1


def print_board(board):
    symbols = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
    print()
    for row in board:
        print(" ".join(symbols[cell] for cell in row))
    print("-" * (COLS * 2 - 1))
    print(" ". join(str(c) for c in range(COLS)))
    print()
