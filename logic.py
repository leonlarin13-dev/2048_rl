import random

def start_game():

    # declaring an empty list then
    # appending 4 list each with four
    # elements as 0.
    mat =[]
    for i in range(4):
        mat.append([0] * 4)

    add_new_2(mat)
    return mat

def findEmpty(mat):
    """Finds the first empty (0) cell in the grid."""
    for i in range(4):
        for j in range(4):
            if mat[i][j] == 0:
                return i, j  # Return the first found empty cell
    return None, None  # No empty cells left

# function to add a new 2 in
# grid at any random empty cell
def add_new_2(mat):
    """Adds a new '2' in a random empty cell in the grid."""
    empty_cells = [(r, c) for r in range(4) for c in range(4) if mat[r][c] == 0]
    if not empty_cells:
        return  # No empty space left
    r, c = random.choice(empty_cells)
    mat[r][c] = 2

# function to get the current
# state of game
def get_current_state(mat):
    """Returns 'WON', 'GAME NOT OVER', or 'LOST' based on the current board state."""
    size = len(mat)

    # Check for empty cells
    for row in mat:
        if 0 in row:
            return 'GAME NOT OVER'

    # Check for a 2048 tile
    for row in mat:
        if 2048 in row:
            return 'WON'


    # Check for possible merges (horizontal and vertical)
    for i in range(size):
        for j in range(size):
            if j + 1 < size and mat[i][j] == mat[i][j + 1]:
                return 'GAME NOT OVER'
            if i + 1 < size and mat[i][j] == mat[i + 1][j]:
                return 'GAME NOT OVER'

    # No moves left
    return 'LOST'


# all the functions defined below
# are for left swap initially.

# function to compress the grid
# after every step before and
# after merging cells.
def compress(mat):
    """Compresses the matrix to the left by removing zeros between numbers."""
    size = len(mat)
    changed = False
    new_mat = []

    for row in mat:
        new_row = [num for num in row if num != 0]  # remove zeros
        new_row += [0] * (size - len(new_row))      # pad with zeros to the right
        if new_row != row:
            changed = True
        new_mat.append(new_row)

    return new_mat, changed


# function to merge the cells
# in matrix after compressing
def merge(mat):
    """Merges tiles in the matrix row-wise to the left."""
    size = len(mat)
    changed = False
    merge_reward = 0

    for i in range(size):
        for j in range(size - 1):
            if mat[i][j] != 0 and mat[i][j] == mat[i][j + 1]:
                mat[i][j] *= 2
                mat[i][j + 1] = 0
                merge_reward += mat[i][j]
                changed = True

    return mat, changed, merge_reward


# function to reverse the matrix
# means reversing the content of
# each row (reversing the sequence)
def reverse(mat):
    new_mat =[]
    for i in range(4):
        new_mat.append([])
        for j in range(4):
            new_mat[i].append(mat[i][3 - j])
    return new_mat

# function to get the transpose
# of matrix means interchanging
# rows and column
def transpose(mat):
    new_mat = []
    for i in range(4):
        new_mat.append([])
        for j in range(4):
            new_mat[i].append(mat[j][i])
    return new_mat

def move_left(grid):
    new_grid, changed1 = compress(grid)
    new_grid, changed2, merge_reward = merge(new_grid)  # updated line
    changed = changed1 or changed2
    new_grid, _ = compress(new_grid)
    return new_grid, changed, merge_reward

def move_right(grid):
    new_grid = reverse(grid)
    new_grid, changed, merge_reward = move_left(new_grid)
    new_grid = reverse(new_grid)
    return new_grid, changed, merge_reward

def move_up(grid):
    new_grid = transpose(grid)
    new_grid, changed, merge_reward = move_left(new_grid)
    new_grid = transpose(new_grid)
    return new_grid, changed, merge_reward

def move_down(grid):
    new_grid = transpose(grid)
    new_grid, changed, merge_reward = move_right(new_grid)
    new_grid = transpose(new_grid)
    return new_grid, changed, merge_reward

# this file only contains all the logic
# functions to be called in main function
# present in the other file

LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

def get_valid_moves(grid):
    """Returns a list of valid directions [0-3] where a move changes the board."""
    valid_moves = []
    for direction in [LEFT, UP, RIGHT, DOWN]:
        new_grid, changed, _ = apply_move(grid, direction)
        if changed:
            valid_moves.append(direction)
    return valid_moves

def apply_move(grid, direction):
    """Applies the move in the given direction. Returns (new_grid, changed, reward)."""
    if direction == LEFT:
        return move_left(grid)
    elif direction == RIGHT:
        return move_right(grid)
    elif direction == UP:
        return move_up(grid)
    elif direction == DOWN:
        return move_down(grid)
    else:
        raise ValueError("Invalid direction: must be 0 (left), 1 (up), 2 (right), or 3 (down)")


#####     not used     #####
def apply_valid_move(grid, direction):
    """
    Applies a move only if it changes the board.
    Returns: (new_grid, reward, move_applied).
    """
    new_grid, changed, reward = apply_move(grid, direction)
    if not changed:
        return grid, 0, False
    return new_grid, reward, True



#####     not used     #####
def get_all_valid_next_states(grid):
    """
    Returns a list of tuples: (direction, resulting_grid, reward) for all valid moves.
    Useful for planning agents (MCC, expectimax, etc.)
    """
    valid_states = []
    for direction in [LEFT, UP, RIGHT, DOWN]:
        new_grid, changed, reward = apply_move(grid, direction)
        if changed:
            valid_states.append((direction, new_grid, reward))
    return valid_states