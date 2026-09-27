from gridworld import SimpleGridWorldEnv
import numpy as np

env = SimpleGridWorldEnv(size=5, agent_loc=(0,0), n_goals=1, n_pits=2)
#vi_model = ValueIteration()

terminated = False
env.render()  # show the actual starting position before the first step
while not terminated:
    action = np.random.choice(['U', 'D', 'L', 'R'])
    
    obs_state, reward, terminated = env.step(action)

    env.render()

env.wait_for_keypress()
env.close()