import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
    
def attacking_pairs(state):
    N = len(state)
    num_att_pairs = 0
    for queen1_col in range(N):
        for queen2_col in range(queen1_col, N):
            if queen1_col != queen2_col:
                queen1_row = state[queen1_col]
                queen2_row = state[queen2_col]
                
                attack_cond = ( queen1_row == queen2_row or  # same row
                                queen1_col == queen2_col or  # same column
                                abs(queen1_row - queen2_row) == abs(queen1_col - queen2_col) ) # check diagonals
                
                if attack_cond:
                    num_att_pairs += 1
    
    return num_att_pairs
    
def get_neighbor(state):
    N = len(state)
    neighbor_state = copy.deepcopy(state)
    
    # randomly pick a queen 
    idx_queen = np.random.randint(low=0, high=N)
    
    # randomly change selected queen's position
    new_row_pos = np.random.randint(low=0, high=N)
    
    neighbor_state[idx_queen] = new_row_pos
    
    return neighbor_state

"""
    Implements simulated annealing for n-queens.
"""
def simulated_annealing(initial_state, initial_temp=1000, decay=0.99):
    # finish implementing it
    pass
    
if __name__ == "__main__":
    import numpy as np
    
    # set random seed for reproducibility
    rand_seed=735122311
    np.random.seed(rand_seed)
    
    # helper function to generate random N-queens state
    def gen_rand_nqueen(N=8):
        return list(np.random.randint(low=0, high=N, size=N))
    
    # experiment with N=4
    initial_state = gen_rand_nqueen(N=4)
    final_state, iters = simulated_annealing(initial_state)
    print(f'N={4}. Starting state attacking_pairs {attacking_pairs(initial_state)}')
    print(f'Took {iters} iters. Ending state attacking_pairs {attacking_pairs(final_state)}')
    print('-------------------------------------------------------------------------')
    
    # experiment with N=8
    initial_state = gen_rand_nqueen(N=8)
    final_state, iters = simulated_annealing(initial_state)
    print(f'N={8}. Starting state attacking_pairs {attacking_pairs(initial_state)}')
    print(f'Took {iters} iters. Ending state attacking_pairs {attacking_pairs(final_state)}')
    print('-------------------------------------------------------------------------')
    
    # experiment with N=16
    initial_state = gen_rand_nqueen(N=16)
    final_state, iters = simulated_annealing(initial_state)
    print(f'N={16}. Starting state attacking_pairs {attacking_pairs(initial_state)}')
    print(f'Took {iters} iters. Ending state attacking_pairs {attacking_pairs(final_state)}')
    print('-------------------------------------------------------------------------')