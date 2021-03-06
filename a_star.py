"""
24 PUZZLE
---------
Given a 5 by 5 board with 24 tiles (each tile is numbered from 1 to 24)
and an empty space (denoted by 0 here), move the tiles so that the
tiles match the user defined end-state.

This is an attempt at solving the 24 puzzle using A-star and Manhattan
heutistic.

EXAMPLE:
    > INITIAL STATE
     1  2  3  4  5
     6  7  8  9 10
    11 12 13 14 15
    16 17  0 19 20
    21 22 23 24 18
    
    > Moved DOWN ↓ from (3, 2) to position (4, 2)
     1  2  3  4  5
     6  7  8  9 10
    11 12 13 14 15
    16 17 23 19 20
    21 22  0 24 18
    
    ...
    
    > FINAL STATE
     1  2  3  4  5
     6  7  8  9 10
    11 12 13 14 15
    16 17 18 19 20
    21 22 23 24  0

    ---------------------------------------------
                    SUMMARY                   
    > MOVES MADE: 11
    > NODES VISITED: 70
    > SECONDS ELAPSED: 0.062
    ---------------------------------------------
"""

import numpy as np, pandas as pd
from queue import PriorityQueue
import random, sys, time

# For console output only
DASH = "-" * 45


# Set up puzzle
class PuzzleBase:
    def __init__(self, state, final, level, parent=None):
        self.__state = state
        self.__final = final
        self.__level = level
        self.__heuristic_score = level
        self.__parent = parent
        self.calculate_fitness()

    def __hash__(self):
        return hash(str(self.__state))

    def __lt__(self, other):
        return self.__heuristic_score < other.__heuristic_score

    def __eq__(self, other):
        return self.__heuristic_score == other.__heuristic_score

    def __gt__(self, other):
        return self.__heuristic_score > other.__heuristic_score

    def get_state(self):
        return self.__state

    def get_score(self):
        return self.__heuristic_score

    def get_level(self):
        return self.__level

    def get_parent(self):
        return self.__parent

    def calculate_fitness(self):
        # Iterate through each tile on the board
        # 0 denotes "empty state"
        for current_tile in self.__state:
            # From the value of the current tile, assign its current index
            current_index = self.__state.index(current_tile)
            # From the value of the current tile, assign index of where it
            # should end
            final_index = self.__final.index(current_tile)
            # Get the current tile position relative to the board
            cur_i, cur_j = current_index // int(
                np.sqrt(len(self.__state))
            ), current_index % int(np.sqrt(len(self.__state)))
            # Get the end tile position relative to the board
            goal_i, goal_j = final_index // int(
                np.sqrt(len(self.__state))
            ), final_index % int(np.sqrt(len(self.__state)))
            # Calculate Manhattan distance between two points
            self.__heuristic_score += self.calculate_manhattan(
                cur_i, cur_j, goal_i, goal_j
            )

    def calculate_manhattan(self, x1, y1, x2, y2):
        """Sum the moves each tile in current state will take to get to gaol state"""
        return abs(x1 - x2) + abs(y1 - y2)


# Define puzzle mechanics
class PuzzleMechanics:
    """Set up base rules for an n x n puzzle"""

    def __init__(self, initial, final, max_tries=999_999):
        self.__initial = initial  # Initial board state
        self.__final = final  # Final board state
        self.__MAX = 100_000  # Size of priority queue
        self.__max_tries = max_tries  # Number of iterations/tries before giving up
        self.__path = []  # Where in the search tree
        self.__summary = ""  # Console output collector

    def set_max_tries(self, max_tries):
        self.__max_tries = max_tries

    def get_path(self):
        return self.__path

    def get_summary(self):
        return self.__summary

    # Use A-star to solve puzzle
    def solve_a_star(self):
        # Define the legal moves for a tile (up, down, etc.)
        x_axis = [1, 0, -1, 0]
        y_axis = [0, 1, 0, -1]

        # Initialise tracking vars
        level = 0
        visited_nodes = set()

        # Start timer
        start_time = time.process_time()

        # Instantiate the priority queue
        nodes = PriorityQueue(self.__MAX)
        # Instantiate the game object
        init_node = PuzzleBase(
            self.__initial.flatten().tolist(),
            self.__final.flatten().tolist(),
            level,
            parent=None,
        )
        # Put items in the queue
        nodes.put(init_node)

        # Sum of visited nodes
        total_visited_nodes = 0
        # Continue if queue size is not None and total_visited_nodes is lte
        # user-defined iterations
        while nodes.qsize() and total_visited_nodes <= self.__max_tries:
            total_visited_nodes += 1

            # Get the position of the blank tile
            cur_node = nodes.get()
            cur_state = cur_node.get_state()

            # Add node to list of visited nodes
            if str(cur_state) in visited_nodes:
                continue
            visited_nodes.add(str(cur_state))

            # When the tiles match the user-defined end state, stop searching
            if cur_state == self.__final.flatten().tolist():
                self.__summary = str(
                    f"> MOVES MADE: {str(cur_node.get_level())}\n"
                    f"> NODES VISITED: {str(total_visited_nodes)}\n"
                    f"> SECONDS ELAPSED: {str(np.round(time.process_time() - start_time, 4))}\n"
                )
                while cur_node.get_parent():
                    self.__path.append(cur_node)
                    cur_node = cur_node.get_parent()
                break

            # Get the current index of "0", which denotes the puzzle's empty state
            empty_tile = cur_state.index(0)
            i, j = (
                empty_tile // self.__final.shape[0],
                empty_tile % self.__final.shape[0],
            )

            # Get current position against final position
            cur_state = np.array(cur_state).reshape(
                self.__final.shape[0], self.__final.shape[0]
            )
            # Iterate through the legal moves for each tile
            for x, y in zip(x_axis, y_axis):
                new_state = np.array(cur_state)
                if (
                    i + x >= 0
                    and i + x < self.__final.shape[0]
                    and j + y >= 0
                    and j + y < self.__final.shape[0]
                ):
                    new_state[i, j], new_state[i + x, j + y] = (
                        new_state[i + x, j + y],
                        new_state[i, j],
                    )
                    puzzle_instance = PuzzleBase(
                        new_state.flatten().tolist(),
                        self.__final.flatten().tolist(),
                        cur_node.get_level() + 1,
                        cur_node,
                    )
                    # Add nodes to array of visited nodes
                    if str(puzzle_instance.get_state()) not in visited_nodes:
                        nodes.put(puzzle_instance)

        # Break out of loop when user-defined max_tries is reached
        if total_visited_nodes > self.__max_tries:
            print(
                "! Puzzle either is impossible to move into final state OR max iterations reached"
            )
        return self.__path


def A_star(initial, final, max_tries):
    """Implement puzzle rules and A-star together and define console outputs"""
    solver = PuzzleMechanics(initial, final, max_tries)
    path = solver.solve_a_star()

    # Get the index of the blank space (i.e., 0)
    init_idx = initial.flatten().tolist().index(0)
    init_i, init_j = init_idx // final.shape[0], init_idx % final.shape[0]
    
    print(DASH)
    print("{:*^45}".format(" INITIAL STATE "))
    print(DASH)
    print(
        pd.DataFrame([initial[i, :] for i in range(final.shape[0])]).to_string(
            header=False, index=False
        )
    )
    print()
    # Display current move of the "blank (i.e., 0)" space
    for node in reversed(path):
        current_index = node.get_state().index(0)
        cur_i, cur_j = current_index // final.shape[0], current_index % final.shape[0]

        new_i, new_j = cur_i - init_i, cur_j - init_j
        if new_j == 0 and new_i == -1:
            print(
                "> Moved UP ↑ from "
                + str((init_i, init_j))
                + " to position "
                + str((cur_i, cur_j))
            )
        elif new_j == 0 and new_i == 1:
            print(
                "> Moved DOWN ↓ from "
                + str((init_i, init_j))
                + " to position "
                + str((cur_i, cur_j))
            )
        elif new_i == 0 and new_j == 1:
            print(
                "> Moved RIGHT → from "
                + str((init_i, init_j))
                + " to position "
                + str((cur_i, cur_j))
            )
        else:
            print(
                "> Moved LEFT ← from "
                + str((init_i, init_j))
                + " to position "
                + str((cur_i, cur_j))
            )
        # print(
        #     f"\n"
        #     f"MANHATTAN HEURISTIC\n"
        #     f"> DISTANCE: {str(node.get_score() - node.get_level())}\n"
        #     f"> LEVEL: {str(node.get_level())}\n"
        # )

        init_i, init_j = cur_i, cur_j

        print(
            pd.DataFrame(
                [
                    np.array(node.get_state()).reshape(final.shape[0], final.shape[0])[
                        i, :
                    ]
                    for i in range(final.shape[0])
                ]
            ).to_string(header=False, index=False)
        )
        print()
    print(DASH)
    print("{: ^45}".format(" SUMMARY "))
    print(solver.get_summary())
    print(DASH)


# DRIVER
def main(argv):
    """
    Get user input and implement A-star with Manhattan heuristic to solve
    n-puzzle

    @param max_tries : The max tries before stopping the search
    @param n         : The size of the board
    @param initial   : The starting state of the puzzle board
    @param final     : The ending state of the puzzle board
    """
    max_tries = 100_000  # Number of iterations before program termination
    n = 5  # n x n size of puzzle board

    while True:
        # Create starting array
        # initial = " ".join(str(elem) for elem in random.sample(range(25), 25))
        # initial = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 0 21 22 23 24 20"
        initial = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 0 19 20 21 22 23 24 18"
        initial = initial.split()
        initial = [int(initial[i]) for i in range(len(initial))]

        # Create the array to end at
        final = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 0"
        final = final.split()
        final = [int(final[i]) for i in range(len(final))]

        # Chck that the length of the user-defined input and output match each
        # other and fits the board
        if len(final) == len(initial) and len(final) == n ** 2:
            break

    # Shape the initial and final arrays into n x n matrices
    initial = np.array(initial).reshape(n, n)
    final = np.array(final).reshape(n, n)

    # Call A-star function
    A_star(initial, final, max_tries)


if __name__ == "__main__":
    main(sys.argv[1:])
