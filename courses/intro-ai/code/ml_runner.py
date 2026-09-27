"""
Naive fitted-Q learning on SimpleGridWorldEnv.

This is the simplest possible deep-Q approach (Riedmiller 2005-style).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # silences the oneDNN line specifically

from gridworld import SimpleGridWorldEnv
import numpy as np
import keras
from keras import layers
from keras import ops
import tensorflow as tf

ACTIONS = ['U', 'D', 'L', 'R']
action_dict = {
    'U': (1, 0, 0, 0),
    'D': (0, 1, 0, 0),
    'L': (0, 0, 1, 0),
    'R': (0, 0, 0, 1),
}
num_actions = 4
input_shape = (2,)
EPSILON = 0.1


def dqn_loss_fn(y_true, y_pred):
    # y_true shape: (batch, num_actions + 1)
    # y_pred shape: (batch, num_actions) = network's Q(s, a) for all a.
    y_label = y_true[:, :1]          # TD target label
    actions = y_true[:, 1:]          # one-hot of the action that was taken

    # we multiply the Q(s, a) y_pred for all actions by the one-hot of the action that was taken
    # and then sum over the actions to get the Q(s, a) for the action that was taken
    q_taken = tf.math.reduce_sum(y_pred * actions, axis=-1, keepdims=True)
    return ops.mean(ops.square(q_taken - y_label))


dqn_model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    keras.layers.Dense(num_actions, activation="linear"),
])
dqn_model.compile(optimizer=keras.optimizers.Adam(), loss=dqn_loss_fn)

np.random.seed(11111)
grid_size = 8
env = SimpleGridWorldEnv(size=grid_size, agent_loc=(0, 0), n_goals=1, n_pits=6)

n_episodes = 500
gamma = 0.9
for epi in range(n_episodes):
    env.reset()
    terminated = False
    iter = 0
    while not terminated:
        current_state = env.agent_loc

        # epsilon-greedy behavior policy (still off-policy update below)
        if np.random.rand() < EPSILON:
            action = np.random.choice(ACTIONS)
        else:
            q_pred = dqn_model.predict(np.array([current_state]), verbose=0)[0, :]
            action = ACTIONS[np.argmax(q_pred)]

        next_state, reward, terminated = env.step(action)

        q_sdash_pred = dqn_model.predict(np.array([next_state]), verbose=0)[0, :]
        q_sdash_pred = q_sdash_pred.max()
        y_label = reward + gamma * q_sdash_pred

        x = np.array([current_state])
        y = np.array([(y_label,) + action_dict[action]])
        loss_hist = dqn_model.train_on_batch(x=x, y=y)

        iter += 1
    print(f'episode #{epi} done loss {loss_hist}')


env.reset()
terminated = False
n_steps = 0
env.render()  # show the actual starting position before the first step
while not terminated:
    current_state = env.agent_loc
    action = dqn_model.predict(np.array([current_state]), verbose=0)[0, :]
    action = ACTIONS[np.argmax(action)]

    next_state, reward, terminated = env.step(action)

    env.render()

    n_steps += 1

env.wait_for_keypress()
env.close()
