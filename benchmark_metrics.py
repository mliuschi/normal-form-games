import torch
import numpy as np
import nashpy as nash
import pickle
from train import total_variation

# output and target and both vectors - not batched
def cross_entropy(output, target, eps=1e-2):
    return -torch.sum(torch.log(output + eps) * target)

if __name__ == '__main__':
    with open("data/two_action_data_test.p", "rb") as file:
        two_action_data = pickle.load(file)
        two_action_x = two_action_data['x']
        two_action_y = two_action_data['y']
    with open("data/three_action_data_test.p", "rb") as file:
        three_action_data = pickle.load(file)
        three_action_x = three_action_data['x']
        three_action_y = three_action_data['y']

    test_ce_nash = 0
    test_tv_nash = 0
    test_ce_random = 0
    test_tv_random = 0

    for x,y in zip(two_action_x, two_action_y):
        y = torch.tensor(y).float()
        game = nash.Game(x[0], x[1])
        equil = game.support_enumeration()
        equil_sum_tv = 0
        equil_sum_ce = 0
        num_equil = 0
        for eq in equil:
            eq = eq[0] # take row player's equilibrium
            out = torch.tensor(eq).float()
            equil_sum_tv += total_variation(out, y)
            equil_sum_ce += cross_entropy(out, y)
            num_equil += 1
        
        test_ce_nash += equil_sum_ce / num_equil
        test_tv_nash += equil_sum_tv / num_equil

        random = torch.ones(len(y)) / len(y)
        test_ce_random += cross_entropy(random, y)
        test_tv_random += total_variation(random, y)

    for x,y in zip(three_action_x, three_action_y):
        y = torch.tensor(y).float()
        game = nash.Game(x[0], x[1])
        equil = game.support_enumeration()
        equil_sum_tv = 0
        equil_sum_ce = 0
        num_equil = 0
        for eq in equil:
            eq = eq[0] # take row player's equilibrium
            out = torch.tensor(eq).float()
            equil_sum_tv += total_variation(out, y)
            equil_sum_ce += cross_entropy(out, y)
            num_equil += 1
        
        test_ce_nash += equil_sum_ce / num_equil
        test_tv_nash += equil_sum_tv / num_equil

        random = torch.ones(len(y)) / len(y)
        test_ce_random += cross_entropy(random, y)
        test_tv_random += total_variation(random, y)

    test_ce_nash /= (len(two_action_x) + len(three_action_x))
    test_tv_nash /= (len(two_action_x) + len(three_action_x))
    test_ce_random /= (len(two_action_x) + len(three_action_x))
    test_tv_random /= (len(two_action_x) + len(three_action_x))

    print("Final results:")
    print("Cross-entropy Nash:", test_ce_nash)
    print("Total variation Nash:", test_tv_nash)
    print("Cross-entropy random guessing:", test_ce_random)
    print("Total variation random guessing:", test_tv_random)