"""
Payoff matrices from "Comparing Theories of One-Shot Play Out of Treatment"
"""

import numpy as np
import pandas as pd
import pickle

hawk_dove_params = [(1,0), (2,0), (3,0), (5,0), (10,0),
                    (3,2), (5,2), (10,2), (10,3), (10,5)]

match_pennies_params = [(1, True), (2, True), (3, True), (5, True), (10, True),
                        (1, False), (2, False), (3, False), (5, False), (10, False)]

hawk_middle_dove_params = [(2,0), (4,0), (6,0), (8,0), (10,0),
                           (6,3), (8,3), (10,3), (10,4), (10,6)]

rsp_params = [(1, True), (2, True), (3, True), (5, True), (10, True),
              (1, False), (2, False), (3, False), (5, False), (10, False)]

def idx_to_softmax(idx, dimension):
    x = np.zeros((dimension,))
    x[idx] = 1.
    return x

def hawk_dove(x, y):
    row = np.array([[0, x], 
                    [1, y]])
    col = np.array([[0, 1], 
                    [x, y]])
    
    # Normalize
    row = (row - np.min(row)) / (np.max(row) - np.min(row))
    col = (col - np.min(col)) / (np.max(col) - np.min(col))

    return np.stack([row, col], axis=0)

def match_pennies(z, row_player):
    row = np.array([[z, 0], 
                    [0, 1]])
    col = np.array([[0, 1], 
                    [1, 0]])
    
    # Normalize
    row = (row - np.min(row)) / (np.max(row) - np.min(row))
    col = (col - np.min(col)) / (np.max(col) - np.min(col))

    if row_player:
        return np.stack([row, col], axis=0)
    else:
        return np.stack([col.T, row.T], axis=0)

def hawk_middle_dove(a, b):
    row = np.array([[0, 1, a], 
                    [1.5, 0, 0.75 * a], 
                    [2, 1, b]])
    col = np.array([[0, 1.5, 2], 
                    [1, 0, 1], 
                    [a, 0.75 * a, b]])
    
    # Normalize
    row = (row - np.min(row)) / (np.max(row) - np.min(row))
    col = (col - np.min(col)) / (np.max(col) - np.min(col))

    return np.stack([row, col], axis=0)

def rock_scissors_paper(z, row_player):
    row = np.array([[0.5, z, 0], 
                    [0, 0.5, 1], 
                    [1, 0, 0.5]])
    col = np.array([[0.5, 0, 1], 
                    [1, 0.5, 0], 
                    [0, 1, 0.5]])
    
    # Normalize
    row = (row - np.min(row)) / (np.max(row) - np.min(row))
    col = (col - np.min(col)) / (np.max(col) - np.min(col))

    if row_player:
        return np.stack([row, col], axis=0)
    else:
        return np.stack([col.T, row.T], axis=0)
    
if __name__ == '__main__':
    two_action = pd.read_csv('two_action_games.csv').to_numpy()[:,1:]
    three_action = pd.read_csv('three_action_games.csv').to_numpy()[:,1:]

    two_action_data = {'x': [], 'y': []}
    three_action_data = {'x': [], 'y': []}

    for (game, choice) in two_action:
        if 'hdg' in game:
            game_type = int(game[-1]) - 1 # - 1 to zero-index
            processed_choice = 1 - choice # convert the standard used in the dataset to valid index
            two_action_data['x'].append(hawk_dove(*hawk_dove_params[game_type]))
            two_action_data['y'].append(idx_to_softmax(processed_choice, 2))
        elif 'mp' in game:
            game_type = int(game[-1]) - 1 # - 1 to zero-index
            processed_choice = 1 - choice # convert the standard used in the dataset to valid index
            two_action_data['x'].append(match_pennies(*match_pennies_params[game_type]))
            two_action_data['y'].append(idx_to_softmax(processed_choice, 2))
        else:
            raise NotImplementedError("Invalid game name!")
        
    for (game, choice) in three_action:
        if 'ac' in game:
            game_type = int(game[-1]) - 1 # - 1 to zero-index
            processed_choice = choice - 1 # convert the standard used in the dataset to valid index
            three_action_data['x'].append(hawk_middle_dove(*hawk_middle_dove_params[game_type]))
            three_action_data['y'].append(idx_to_softmax(processed_choice, 3))
        elif 'rsp' in game:
            game_type = int(game[-1]) - 1 # - 1 to zero-index
            processed_choice = choice - 1 # convert the standard used in the dataset to valid index
            three_action_data['x'].append(rock_scissors_paper(*rsp_params[game_type]))
            three_action_data['y'].append(idx_to_softmax(processed_choice, 3))
        else:
            raise NotImplementedError("Invalid game name!")

    # Save data with pickle
    with open("two_action_data.p", "wb") as file:
        pickle.dump(two_action_data, file)

    with open("three_action_data.p", "wb") as file:
        pickle.dump(three_action_data, file)