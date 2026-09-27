from gridworld import SimpleGridWorldEnv
import numpy as np

# tweak these to explore reward shaping 
GOAL_REWARD = 50
PIT_REWARD  = -100
STEP_REWARD = -1
GAMMA       = 0.9

np.random.seed(32131313)

env = SimpleGridWorldEnv(
    size=8, agent_loc=(0,7), n_goals=2, n_pits=4,
    step_reward=STEP_REWARD, goal_reward=GOAL_REWARD, pit_reward=PIT_REWARD,
)

V, policy = env.do_value_iteration(gamma=GAMMA, render=True, font_scale=22)

env.wait_for_keypress()
env.close()


terminated = False
obs_state = env.agent_loc
env.render()  # show the actual starting position before the first step
while not terminated:
    action = policy[obs_state]

    obs_state, reward, terminated = env.step(action)

    env.render()

env.wait_for_keypress()
env.close()


# Goal at (10, 6); 3 fixed pits flank N/E/S so the only safe approach is
# from the west. Agent starts on the same row at (10, 14), so it must
# detour above or below to reach the goal from the open side. The other
# 27 pits are scattered randomly across the rest of the grid.
env = SimpleGridWorldEnv(
    size=15, agent_loc=(10,14), n_goals=1, n_pits=30,
    goal_locs=[(10, 6)],
    pit_locs=[(9, 6), (10, 7), (11, 6)],
    step_reward=STEP_REWARD, goal_reward=GOAL_REWARD, pit_reward=PIT_REWARD,
)

V, policy = env.do_value_iteration(gamma=GAMMA, render=True, font_scale=11)

env.wait_for_keypress()
env.close()

terminated = False
obs_state = env.agent_loc
env.render()  # show the actual starting position before the first step
while not terminated:
    action = policy[obs_state]

    obs_state, reward, terminated = env.step(action)

    env.render()

env.wait_for_keypress()
env.close()