import sys
import os
import random
from logic import *


def print_board(mat):
    os.system('cls' if os.name == 'nt' else 'clear')  
    print("2048 Game - Use W (Up), A (Left), S (Down), D (Right). Press Q to quit.\n")
    for row in mat:
        print("+------" * 4 + "+")
        print("".join(f"|{str(num).center(6) if num != 0 else ' '.center(6)}" for num in row) + "|")
    print("+------" * 4 + "+\n")

def main():
    mat = start_game()
    print_board(mat)

    while True:
        move = input("Your move (W/A/S/D): ").strip().upper()
        if move not in ['W', 'A', 'S', 'D', 'Q']:
            print("Invalid input! Please enter W, A, S, D or Q to quit.")
            continue
        if move == 'Q':
            print("Thanks for playing!")
            break

        if move == 'W':
            mat, changed, _ = move_up(mat)
        elif move == 'A':
            mat, changed, _ = move_left(mat)
        elif move == 'S':
            mat, changed, _ = move_down(mat)
        elif move == 'D':
            mat, changed, _ = move_right(mat)

        if changed:
            add_new_2(mat)
            print_board(mat)
        else:
            print("No tiles moved. Try a different direction.")

        state = get_current_state(mat)
        if state == 'WON':
            print_board(mat)
            print("Congratulations! You reached 2048 and won the game!")
            break
        elif state == 'LOST':
            print_board(mat)
            print("Game over! No more moves left.")
            break
        # if 'GAME NOT OVER', continue playing

if __name__ == "__main__":
    main()
