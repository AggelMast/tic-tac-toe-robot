"""Stateful Tic-Tac-Toe opponent.

This module maintains board state across moves, detects winning and tie positions,
and chooses the best move for the robot opponent using minimax.

Usage from another program:
- Start this script once and send moves one per line on stdin.
- Each move is a number from 1 to 9:
    1 2 3
    4 5 6
    7 8 9
- The robot returns its move as a number from 1 to 9.
- Send "reset" to start a new game.
"""

from __future__ import annotations

WINNING_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]

USER_SYMBOL = "X"
ROBOT_SYMBOL = "O"
EMPTY_SYMBOL = " "

board: list[str] = [EMPTY_SYMBOL] * 9

def reset_board() -> None:
    """Reset the board to an empty game state."""
    global board
    board = [EMPTY_SYMBOL] * 9


def board_to_string() -> str:
    """Return the current board as a single string for easy storage or display."""
    return "".join(board)


def get_available_moves(state: list[str]) -> list[int]:
    return [i for i, value in enumerate(state) if value == EMPTY_SYMBOL]


def is_winner(state: list[str], symbol: str) -> bool:
    return any(all(state[pos] == symbol for pos in line) for line in WINNING_LINES)


def is_tie(state: list[str]) -> bool:
    return EMPTY_SYMBOL not in state and not is_winner(state, USER_SYMBOL) and not is_winner(state, ROBOT_SYMBOL)


def evaluate_state(state: list[str]) -> int:
    if is_winner(state, ROBOT_SYMBOL):
        return 10
    if is_winner(state, USER_SYMBOL):
        return -10
    return 0


def minimax(state: list[str], depth: int, is_robot_turn: bool, alpha: int, beta: int) -> int:
    score = evaluate_state(state)
    if score != 0 or is_tie(state):
        return score

    if is_robot_turn:
        best = -999
        for move in get_available_moves(state):
            state[move] = ROBOT_SYMBOL
            value = minimax(state, depth + 1, False, alpha, beta)
            state[move] = EMPTY_SYMBOL
            best = max(best, value)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best

    best = 999
    for move in get_available_moves(state):
        state[move] = USER_SYMBOL
        value = minimax(state, depth + 1, True, alpha, beta)
        state[move] = EMPTY_SYMBOL
        best = min(best, value)
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def find_best_robot_move() -> int | None:
    """Return the best robot move index (0-8), or None if no moves remain."""
    if is_winner(board, USER_SYMBOL) or is_winner(board, ROBOT_SYMBOL) or is_tie(board):
        return None

    best_value = -999
    best_move: int | None = None
    for move in get_available_moves(board):
        board[move] = ROBOT_SYMBOL
        move_value = minimax(board, 0, False, -1000, 1000)
        board[move] = EMPTY_SYMBOL
        if move_value > best_value:
            best_value = move_value
            best_move = move
    return best_move


def apply_move(position: int, symbol: str) -> None:
    if position < 1 or position > 9:
        raise ValueError("Move must be an integer from 1 to 9.")
    index = position - 1
    if board[index] != EMPTY_SYMBOL:
        raise ValueError(f"Position {position} is already occupied.")
    board[index] = symbol


def process_user_move(position: int) -> int | None:
    """Apply the user's move and return the robot's best move as a 1-9 position."""
    apply_move(position, USER_SYMBOL)

    if is_winner(board, USER_SYMBOL) or is_tie(board):
        return None

    best_move = find_best_robot_move()
    if best_move is None:
        return None

    board[best_move] = ROBOT_SYMBOL
    return best_move + 1


def game_status() -> str:
    if is_winner(board, ROBOT_SYMBOL):
        return "robot_wins"
    if is_winner(board, USER_SYMBOL):
        return "user_wins"
    if is_tie(board):
        return "tie"
    return "in_progress"


def get_current_board() -> list[str]:
    """Return a shallow copy of the current board state."""
    return board.copy()


def format_board(state: list[str] | None = None) -> str:
    state = state if state is not None else board
    lines = [
        f" {state[0]} | {state[1]} | {state[2]} ",
        "---+---+---",
        f" {state[3]} | {state[4]} | {state[5]} ",
        "---+---+---",
        f" {state[6]} | {state[7]} | {state[8]} ",
    ]
    return "\n".join(lines)


def _read_input_line() -> str | None:
    try:
        return input().strip()
    except EOFError:
        return None


def _handle_command(command: str) -> None:
    normalized = command.strip().lower()
    if normalized in {"reset", "r"}:
        reset_board()
        print("RESET")
        return

    try:
        user_move = int(command)
    except ValueError:
        print("ERROR: input must be a number from 1 to 9 or 'reset'.")
        return

    try:
        robot_move = process_user_move(user_move)
    except ValueError as error:
        print(f"ERROR: {error}")
        return

    status = game_status()
    if status == "robot_wins":
        print(f"{robot_move if robot_move is not None else '0'} ROBOT_WINS")
    elif status == "user_wins":
        print("USER_WINS")
    elif status == "tie":
        print(f"{robot_move if robot_move is not None else '0'} TIE")
    else:
        print(robot_move)


if __name__ == "__main__":
    print("Tic-Tac-Toe robot ready. Enter moves 1-9, or 'reset' to start a new game.")
    while True:
        line = _read_input_line()
        if line is None:
            break
        if not line:
            continue
        _handle_command(line)

