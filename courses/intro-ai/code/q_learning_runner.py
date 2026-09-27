"""
Tabular Q-learning on SimpleGridWorldEnv.

Same Bellman optimality target as Value Iteration, but model-free: instead
of summing over the known transition probabilities, we *sample* a transition
by acting in the env and bootstrap off the current Q estimate.

    VI update:  V(s)    <- max_a sum_s' p(s'|s,a) [r + gamma V(s')]
    Q-learning: Q(s,a)  <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

Off-policy: behavior is epsilon-greedy, but the target uses max_a' Q(s',a')
regardless of which action we'd actually take next. Swap the max for
Q[s'][a'] (with a' sampled epsilon-greedy) and you have SARSA.
"""
from gridworld import SimpleGridWorldEnv
import numpy as np

# tweak these to explore reward shaping
GOAL_REWARD = 20
PIT_REWARD  = -5
STEP_REWARD = -1

# Q-learning hyperparameters
GAMMA        = 0.9
ALPHA        = 0.1     # learning rate (step size on the TD error)
EPSILON      = 0.2     # exploration probability (fixed; no annealing)
N_EPISODES   = 500
RENDER_EVERY = 50      # render Q-table every N episodes during training

ACTIONS = ['U', 'D', 'L', 'R']

np.random.seed(32131313)

env = SimpleGridWorldEnv(
    size=10, agent_loc=(0, 0), n_goals=1, n_pits=2,
    step_reward=STEP_REWARD, goal_reward=GOAL_REWARD, pit_reward=PIT_REWARD,
)

# Q[s][a]; terminal states stay at 0 (no bootstrap from beyond them)
Q = {s: {a: 0.0 for a in ACTIONS} for s in env.states}


def epsilon_greedy(state, eps):
    if np.random.rand() < eps:
        return np.random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: Q[state][a])


def derive_V_and_policy():
    # V(s) = max_a Q(s, a)
    # policy = argmax. 
    V = {s: max(Q[s].values()) for s in env.states}
    policy = {s: '-' if (s in env.goals or s in env.pits)
              else max(ACTIONS, key=lambda a: Q[s][a])
              for s in env.states}
    return V, policy


for episode in range(N_EPISODES):
    env.reset()
    terminated = False
    while not terminated:
        # initialize starting state
        s = env.agent_loc
        # take epsilon-greedy action
        a = epsilon_greedy(s, EPSILON)
        # take action and get next state, reward, and termination
        s_next, r, terminated = env.step(a)

        # off-policy TD target: bootstraps off greedy action at s_next (max_a' Q(s',a'))
        max_q_next = 0.0 if terminated else max(Q[s_next].values())
        td_target = r + GAMMA * max_q_next
        Q[s][a] = Q[s][a] + ALPHA * (td_target - Q[s][a])

    if (episode + 1) % RENDER_EVERY == 0:
        V, policy = derive_V_and_policy()
        env.render_value_iteration(V, policy, font_scale=32)
        print(f'episode {episode + 1}/{N_EPISODES}')

print('training done. press space to roll out the learned policy.')
V, policy = derive_V_and_policy()
env.render_value_iteration(V, policy, font_scale=32)
env.wait_for_keypress()
env.close()


# greedy rollout with the learned policy
env.reset()
terminated = False
obs_state = env.agent_loc
env.render()  # show the actual starting position before the first step
while not terminated:
    action = policy[obs_state]
    obs_state, reward, terminated = env.step(action)
    env.render()

env.wait_for_keypress()
env.close()
