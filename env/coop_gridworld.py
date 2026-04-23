import numpy as np

class CoopGridWorld:
    def __init__(self, width=7, height=7):
        self.width = width
        self.height = height
        self.n_agents = 2
        self.n_actions = 5
        self.episode_limit = 25
        
        # Grid layout
        # (1,1) is Plate
        self.plate_pos = np.array([1, 1])
        # (3,3) is Door. Wall is x=3, exception is y=3
        self.wall_col = 3
        self.door_y = 3
        # (5,5) is Treasure
        self.treasure_pos = np.array([5, 5])
        
        self.reset()
        
    def reset(self):
        # Starts on left side of the wall
        self.agent_pos = [
            np.array([0, 0]),
            np.array([0, self.height - 1])
        ]
        self.steps = 0
        # Milestone flags — one-time bonuses to prevent reward hacking
        self.plate_bonus_given    = False
        self.door_bonus_given     = False
        return self.get_obs(), self.get_state()

    def get_door_open(self):
        # Door is open if any agent is on the plate
        for pos in self.agent_pos:
            if np.array_equal(pos, self.plate_pos):
                return True
        return False

    def step(self, actions):
        self.steps += 1
        reward = -0.05  # time penalty — encourages agents to act fast
        done = False
        
        door_open = self.get_door_open()
        
        for i, action in enumerate(actions):
            new_pos = np.copy(self.agent_pos[i])
            if action == 0:   # Up
                new_pos[1] += 1
            elif action == 1: # Down
                new_pos[1] -= 1
            elif action == 2: # Left
                new_pos[0] -= 1
            elif action == 3: # Right
                new_pos[0] += 1
            elif action == 4: # Stay
                pass
            
            # Boundary checks
            if new_pos[0] < 0 or new_pos[0] >= self.width or new_pos[1] < 0 or new_pos[1] >= self.height:
                continue # hit boundary, ignore
            
            # Wall and door checks
            if new_pos[0] == self.wall_col:
                if new_pos[1] != self.door_y:
                    continue # hit wall
                else:
                    if not door_open:
                        continue # door is closed
            
            # Valid move
            self.agent_pos[i] = new_pos
        
        # Penalty: agents on the same cell
        if np.array_equal(self.agent_pos[0], self.agent_pos[1]):
            reward -= 0.2
            
        # If an agent steps ON the treasure, success
        for pos in self.agent_pos:
            if np.array_equal(pos, self.treasure_pos):
                reward += 10.0
                done = True
                break
                
        if self.steps >= self.episode_limit:
            done = True

        # One-time milestone bonuses (prevent per-step exploitation)
        if not done:
            # First time plate is activated
            if not self.plate_bonus_given and self.get_door_open():
                reward += 0.5
                self.plate_bonus_given = True
                
            # First time an agent stands in front of the door
            for pos in self.agent_pos:
                if pos[1] == self.door_y and pos[0] == self.wall_col - 1:
                    if not self.door_bonus_given:
                        reward += 0.3
                        self.door_bonus_given = True
                    break
            
        return reward, done, {}

    def get_obs(self):
        """Each agent observes: my_pos, other_pos, door_open, plate_pos, treasure_pos — 9 values."""
        obs = []
        door_open = 1.0 if self.get_door_open() else 0.0
        norm = np.array([self.width, self.height])
        plate_norm  = self.plate_pos    / norm
        treasure_norm = self.treasure_pos / norm
        for i in range(self.n_agents):
            my_pos    = self.agent_pos[i]   / norm
            other_pos = self.agent_pos[1-i] / norm
            # [my_x, my_y, other_x, other_y, door_open, plate_x, plate_y, treasure_x, treasure_y]
            o = np.concatenate([my_pos, other_pos, [door_open], plate_norm, treasure_norm])
            obs.append(o)
        return obs

    def get_state(self):
        """Global state: both agent positions + door_open + plate_pos + treasure_pos — 9 values."""
        door_open = 1.0 if self.get_door_open() else 0.0
        norm = np.array([self.width, self.height])
        p0 = self.agent_pos[0] / norm
        p1 = self.agent_pos[1] / norm
        plate_norm    = self.plate_pos    / norm
        treasure_norm = self.treasure_pos / norm
        return np.concatenate([p0, p1, [door_open], plate_norm, treasure_norm])

    def get_obs_size(self):
        return 9  # [my_x, my_y, other_x, other_y, door_open, plate_x, plate_y, treasure_x, treasure_y]

    def get_state_size(self):
        return 9  # same features in global state

    def get_avail_actions(self):
        return [[1] * self.n_actions for _ in range(self.n_agents)]
