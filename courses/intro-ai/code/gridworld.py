import pygame
import numpy as np


class SimpleGridWorldEnv():
    def __init__(self, size=5, agent_loc=(0,0), n_goals=1, n_pits=2,
                 goal_locs=None, pit_locs=None,
                 step_reward=-1, goal_reward=20, pit_reward=-5,
                 stochastic=False, p_intended=0.8):
        self.render_fps = 2
        self.clock = pygame.time.Clock()
        
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        
        # setup actions
        self._action_to_direction = {
            'U': (-1, 0),
            'D': (+1, 0),
            'L': (0, -1),
            'R': (0, +1),
        }
        
        # setup states
        self.states = []
        for row in range(self.size):
            for col in range(self.size):
                self.states.append( (row, col) )
        
        # setup locations
        self.start_agent_loc = agent_loc
        self.agent_loc = agent_loc
        
        self.goals = []
        valid_states = [s for s in self.states if s != self.agent_loc]
        if goal_locs is not None:
            # Explicit placement: validate each, then claim it.
            for goal_loc in goal_locs:
                goal_loc = tuple(goal_loc)
                if goal_loc not in valid_states:
                    raise ValueError(
                        f"goal_loc {goal_loc} is invalid: must be inside the "
                        f"{size}x{size} grid, not the agent start, and not a "
                        "duplicate."
                    )
                self.goals.append(goal_loc)
                valid_states.remove(goal_loc)
        else:
            for i in range(n_goals):
                rnd_idx = np.random.choice(len(valid_states))
                goal_loc = valid_states[rnd_idx]
                self.goals.append(goal_loc)
                valid_states.remove(goal_loc)
            
        
        self.pits = []
        # Place explicit pits first (if any), then fill the rest randomly up
        # to n_pits. Lets you anchor a few pits for a teaching scenario while
        # still scattering the rest.
        if pit_locs is not None:
            for pit_loc in pit_locs:
                pit_loc = tuple(pit_loc)
                if pit_loc not in valid_states:
                    raise ValueError(
                        f"pit_loc {pit_loc} is invalid: must be inside the "
                        f"{size}x{size} grid, not the agent start, not a "
                        "goal, and not a duplicate."
                    )
                self.pits.append(pit_loc)
                valid_states.remove(pit_loc)
        n_random_pits = max(0, n_pits - len(self.pits))
        for i in range(n_random_pits):
            rnd_idx = np.random.choice(len(valid_states))
            pit_loc = valid_states[rnd_idx]
            self.pits.append(pit_loc)
            valid_states.remove(pit_loc)
        
        # setup rewards
        # Exposed via constructor so we can do reward shaping:
        # e.g. pit_reward=-5 -> agent may walk over a pit to reach goal faster
        # vs. pit_reward=-1000 -> agent learns to go around (catastrophic pit).
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.pit_reward = pit_reward

        self.rewards = {s: step_reward for s in self.states}
        for goal in self.goals:
            self.rewards[goal] = goal_reward
        for pit in self.pits:
            self.rewards[pit] = pit_reward
        
        # setup probability dynamics
        # self.probas[s][a] is a list of (s_dash, reward, probability) outcomes.
        # - deterministic: single outcome with p=1.0.
        # - stochastic: 80/10/10 split (Russell-Norvig style):
        #     p_intended on intended cell, (1-p_intended)/2 each on the
        #     two perpendicular cells. Slips into walls clip and stay.
        self.stochastic = stochastic
        self.p_intended = p_intended

        self.probas = {}
        for s in self.states:
            self.probas[s] = {}
            for a, direction in self._action_to_direction.items():
                intended = self._next_cell(s, direction)
                if stochastic:
                    perp_left = self._next_cell(s, (-direction[1], direction[0]))
                    perp_right = self._next_cell(s, (direction[1], -direction[0]))
                    p_perp = (1.0 - p_intended) / 2.0
                    self.probas[s][a] = [
                        (intended,   self.rewards[intended],   p_intended),
                        (perp_left,  self.rewards[perp_left],  p_perp),
                        (perp_right, self.rewards[perp_right], p_perp),
                    ]
                else:
                    self.probas[s][a] = [(intended, self.rewards[intended], 1.0)]

    def _next_cell(self, s, direction):
        row = min(max(s[0] + direction[0], 0), self.size-1)
        col = min(max(s[1] + direction[1], 0), self.size-1)
        return (row, col)

    def step(self, action):
        if self.agent_loc in self.goals or self.agent_loc in self.pits:
            raise RuntimeError(
                f"step() called from terminal state {self.agent_loc}; "
                "call reset() first."
            )

        outcomes = self.probas[self.agent_loc][action]
        probs = [p for (_, _, p) in outcomes]
        idx = np.random.choice(len(outcomes), p=probs)
        s_dash, reward, _ = outcomes[idx]

        self.agent_loc = s_dash

        terminated = (s_dash in self.goals) or (s_dash in self.pits)

        return s_dash, reward, terminated


    def _draw_goals_and_pits(self, canvas, pix_square_size):
        for goal in self.goals:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * np.array(goal)[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )
        for pit in self.pits:
            pygame.draw.rect(
                canvas,
                (150, 100, 0),
                pygame.Rect(
                    pix_square_size * np.array(pit)[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )

    def _draw_gridlines(self, canvas, pix_square_size):
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

    def render(self):
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (self.window_size / self.size)

        self._draw_goals_and_pits(canvas, pix_square_size)

        # draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.array(self.agent_loc)[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        self._draw_gridlines(canvas, pix_square_size)

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.render_fps)

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def wait_for_keypress(self, key=pygame.K_SPACE):
        # block until the user presses `key` (default: space) 
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == key:
                    return
            pygame.time.wait(50)  # avoid pegging the CPU
            
    def do_value_iteration(self, gamma=0.9, render=True, font_scale=32):
        V = {s: 0 for s in self.states}
        # Goals/pits are terminal -> no real action; '-' is rendered grey.
        policy = {s: '-' if (s in self.goals or s in self.pits) else 'R'
                  for s in self.states}

        if render:
            # Show the initial (all-zero) V-table and wait for the user to
            # press space before starting the sweeps. Lets you explain the
            # env layout in class before the animation kicks off.
            self.render_value_iteration(V, policy, font_scale=font_scale)
            self.wait_for_keypress()

        threshold = 0.000001
        delta_change = 1
        while delta_change > threshold:
            delta_change = 0
            for s in self.states:
                if s in self.goals or s in self.pits:
                    continue
                v_s = V[s]
                
                max_act_val = None
                for a in self._action_to_direction:
                    curr_val = 0.0
                    for (s_dash, r, p) in self.probas[s][a]:
                        curr_val += p * (r + gamma * V[s_dash])

                    if max_act_val is None or curr_val > max_act_val:
                        max_act_val = curr_val
                        policy[s] = a
                
                # update state using max among actions
                V[s] = max_act_val
                delta_change = max(delta_change, abs(v_s - V[s]))
            
            if render:
                self.render_value_iteration(V, policy, font_scale=font_scale)
                
        return V, policy
        
    def render_value_iteration(self, V, policy, fps=2, font_scale=32):
        pygame.init()
        font = pygame.font.Font('freesansbold.ttf', font_scale)
        pygame.display.init()
        self.window = pygame.display.set_mode((3*self.window_size, self.window_size))

        pix_square_size = (self.window_size / self.size)

        # left canvas: env (goals, pits, reward values)
        env_canvas = pygame.Surface((self.window_size, self.window_size))
        env_canvas.fill((255, 255, 255))
        self._draw_goals_and_pits(env_canvas, pix_square_size)
        for s in self.states:
            text = font.render(f'{self.rewards[s]}', True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = pix_square_size * np.array(s)[::-1] + pix_square_size / 2
            env_canvas.blit(text, textRect)

        # middle canvas: V values, color-coded
        vi_canvas = pygame.Surface((self.window_size, self.window_size))
        vi_canvas.fill((255, 255, 255))
        for s in self.states:
            v_s = V[s]
            red_scale = 0
            green_scale = 0
            if v_s > 0:
                green_scale = min(int(100 * v_s / 20), 255)
            else:
                red_scale = min(int(100 * abs(v_s) / 2), 255)
            pygame.draw.rect(
                vi_canvas,
                (red_scale, green_scale, 0),
                pygame.Rect(
                    pix_square_size * np.array(s)[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )
            text = font.render(f'{v_s:.2f}', True, (200, 200, 200))
            textRect = text.get_rect()
            textRect.center = pix_square_size * np.array(s)[::-1] + pix_square_size / 2
            vi_canvas.blit(text, textRect)

        # right canvas: policy arrows
        policy_canvas = pygame.Surface((self.window_size, self.window_size))
        policy_canvas.fill((255, 255, 255))
        color_dict = {'U': (100, 100, 0),
                      'D': (100, 0, 100),
                      'R': (0, 0, 100),
                      'L': (0, 100, 100),
                      '-': (180, 180, 180)}
        for s in self.states:
            action = policy[s]
            text = font.render(f'{action}', True, color_dict[action])
            textRect = text.get_rect()
            textRect.center = pix_square_size * np.array(s)[::-1] + pix_square_size / 2
            policy_canvas.blit(text, textRect)

        for canvas in [env_canvas, vi_canvas, policy_canvas]:
            self._draw_gridlines(canvas, pix_square_size)

        self.window.blit(env_canvas, (0, 0))
        self.window.blit(vi_canvas, (self.window_size, 0))
        self.window.blit(policy_canvas, (2*self.window_size, 0))
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(fps)
        
    def reset(self):
        valid_states = [s for s in self.states if s not in self.goals]
        valid_states = [s for s in valid_states if s not in self.pits]
        rnd_idx = np.random.choice(len(valid_states))
        self.agent_loc = valid_states[rnd_idx]