import numpy as np

class SlidingBlock:
    # state will be a flattened tuple 
    # e.g. state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    # 0 is empty block
    def __init__(self, initial_state):
        self.goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.initial_state = initial_state

    def actions(self, state):
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # convert to 3x3 numpy array for access to features
        np_state = np.array(state).reshape(3,3)

        # get location of empty block
        eb_i, eb_j = np.where(np_state == 0)
        eb_i, eb_j = eb_i[0], eb_j[0]
        
        if eb_i == 0:
            possible_actions.remove('UP')
        if eb_i == 2:
            possible_actions.remove('DOWN')
        if eb_j == 0:
            possible_actions.remove('LEFT')
        if eb_j == 2:
            possible_actions.remove('RIGHT')

        return possible_actions

    def result(self, state, action):
        # convert to 3x3 numpy array for access to features
        np_state = np.array(state).reshape(3,3)
        
        # get location of empty block
        eb_i, eb_j = np.where(np_state == 0)
        eb_i, eb_j = eb_i[0], eb_j[0]
        
        if action == 'UP':
            np_state[eb_i, eb_j] = np_state[eb_i-1, eb_j]
            np_state[eb_i-1, eb_j] = 0
        elif action == 'DOWN':
            np_state[eb_i, eb_j] = np_state[eb_i+1, eb_j]
            np_state[eb_i+1, eb_j] = 0
        elif action == 'LEFT':
            np_state[eb_i, eb_j] = np_state[eb_i, eb_j-1]
            np_state[eb_i, eb_j-1] = 0
        elif action == 'RIGHT':
            np_state[eb_i, eb_j] = np_state[eb_i, eb_j+1]
            np_state[eb_i, eb_j+1] = 0
            
        # convert back to tuple
        next_state = tuple(np_state.flatten())
        return next_state

    def action_cost(self, state1, action, state2):
        return 1

    def is_goal(self, state):
        return state == self.goal_state
        
    """
        h1 heuristic from lecture slides.
        Allow a tile to move anywhere. 
        h1 = number of misplaced tiles
    """
    def h1(self, node):
        curr_state = node.state
        if self.is_goal(curr_state):
            return 0
        else:
            # convert to 3x3 numpy array for access to features
            np_curr_state = np.array(curr_state).reshape(3,3)
            np_goal_state = np.array(self.goal_state).reshape(3,3)
            
            # using numpy features
            unequal_tiles = (np_curr_state != np_goal_state) 
            num_misplaced_tiles = unequal_tiles.sum()
            
            return num_misplaced_tiles
    
    """
        h2 heuristic from lecture slides.
        Allow a tile to move to any adjacent square.
        h2 = sum of distances of tiles from their goal (Manhattan distance)
    """
    def h2(self, node):
        curr_state = node.state
        if self.is_goal(curr_state):
            return 0
        else:
            # convert to 3x3 numpy array for access to features
            np_curr_state = np.array(curr_state).reshape(3,3)
            np_goal_state = np.array(self.goal_state).reshape(3,3)
            
            sum_dist_from_goal = 0
            
            for row in range(3):
                for col in range(3):
                    if (np_goal_state[row,col] != np_curr_state[row,col]): # tile not at goal position
                        misplaced_row, misplaced_col = np.where(np_curr_state == np_goal_state[row,col])
                        misplaced_row = misplaced_row[0]
                        misplaced_col = misplaced_col[0]
                        
                        manhattan_dist = abs(row - misplaced_row) + abs(col - misplaced_col)
                        sum_dist_from_goal += manhattan_dist
                        
            return sum_dist_from_goal
            
    def check_solvability(self, state):
        """ Checks if the given state is solvable """
        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1
        return inversion % 2 == 0