import copy

class GridPainter:
    def __init__(self, N):
        self.N = N

        # setup P matrix
        P = [['u' for _ in range(N)] for _ in range(N)]

        # what the initial state looks like
        self.initial_state = [0, 0, P]
        
    def actions(self, state):
        row = state[0]
        col = state[1]
        P = state[2]
        
        available_actions = ['up', 'down', 'left', 'right', 'blue', 'red']
        
        # handle edge cases
        if row == (self.N-1):
            available_actions.remove('down')
        if row == 0:
            available_actions.remove('up')
        if col == 0:
            available_actions.remove('left')
        if col == (self.N-1):
            available_actions.remove('right')

        # do not allow painting already painted cells
        # this is done to reduce search work, otherwise search takes long
        if P[row][col] != 'u':  # if cell is already painted (blue or red)
            if 'blue' in available_actions:
                available_actions.remove('blue')
            if 'red' in available_actions:
                available_actions.remove('red')
            
        return available_actions
        
    def result(self, state, action):
        # computationally model the environment transitions
        row = state[0]
        col = state[1]
        P = copy.deepcopy(state[2])
        
        available_actions = ['up', 'down', 'left', 'right', 'blue', 'red']
        
        if action == 'up':
            row = row-1
        elif action == 'down':
            row = row+1
        elif action == 'left':
            col = col-1
        elif action == 'right':
            col = col+1
        elif action == 'blue':
            P[row][col] = 'b'
        elif action == 'red':
            P[row][col] = 'r'

        next_state = [row, col, P]

        return next_state
        
    # check that P has half blue and half red
    def is_goal(self, state):
        P = state[2]
        
        blue_count = 0
        red_count = 0

        for row in range(self.N):
            for col in range(self.N):
                if P[row][col] == 'r':
                    red_count += 1
                elif P[row][col] == 'b':
                    blue_count += 1
        
        half_count = (self.N * self.N) / 2
        return (blue_count == half_count) and (red_count == half_count)


if __name__ ==  '__main__':
    gp_example = GridPainter(N=2)
    print(gp_example.initial_state)

    P = [['u', 'u'],
         ['u', 'u']]
    state = [0, 0, P]
    print(gp_example.actions(state))
    print(gp_example.result(state, 'blue'))
    print(gp_example.is_goal(state))
    print()

    
    P = [['b', 'b'],
         ['r', 'b']]
    state = [0, 0, P]
    print(gp_example.is_goal(state)) # should return False

    
    P = [['b', 'r'],
         ['b', 'r']]
    state = [1, 1, P]
    print(gp_example.is_goal(state)) # should return True