import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiAgentMonitoring(gym.Env):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.map_size = cfg.map_size
        self.num_agents = cfg.num_agents
        self.gt_map = self._load_map(cfg.map_data_path)
        self.gt_map_normalized = (self.gt_map - self.gt_map.min()) / (self.gt_map.max() - self.gt_map.min() + 1e-6)
        
        self.action_space = spaces.Box(-1.0, 1.0, (self.num_agents, 2), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.num_agents * 3,), dtype=np.float32)

    def _load_map(self, path):
        with open(path, 'r') as f:
            content = f.read(100)
            delim = ',' if ',' in content else None
        return np.genfromtxt(path, delimiter=delim)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.pos = np.array([self._find_valid_pos() for _ in range(self.num_agents)], dtype=np.float32)
        self.visited = np.zeros(self.map_size)
        self.steps = 0
        
        # Sense telemetry values right at spawning points
        initial_sensed_values = []
        for i in range(self.num_agents):
            ix, iy = np.clip(self.pos[i], [0, 0], [self.map_size[1]-1, self.map_size[0]-1]).astype(int)
            val = np.clip(self.gt_map_normalized[iy, ix] + np.random.normal(0, 0.05), 0, 1)
            initial_sensed_values.append((self.pos[i][0], self.pos[i][1], val))
            self.visited[iy, ix] = 1  # Mark start points as visited

        info = {
            "visited": self.visited,
            "sensed_values": initial_sensed_values
        }
        return self._get_obs(), info

    def _find_valid_pos(self):
        while True:
            y, x = np.random.randint(0, self.map_size[0]), np.random.randint(0, self.map_size[1])
            if self.gt_map_normalized[y, x] >= self.cfg.collision_threshold:
                return [float(x), float(y)]

    def _get_obs(self):
        obs = []
        for p in self.pos:
            ix, iy = np.clip(p, [0, 0], [self.map_size[1]-1, self.map_size[0]-1]).astype(int)
            val = np.clip(self.gt_map_normalized[iy, ix] + np.random.normal(0, 0.05), 0, 1)
            obs.extend([p[0] / self.map_size[1], p[1] / self.map_size[0], val])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        reward = 0
        new_pos = self.pos + action
        sended_values_at_step = []
        
        for i in range(self.num_agents):
            ix, iy = np.clip(new_pos[i], [0, 0], [self.map_size[1]-1, self.map_size[0]-1]).astype(int)
            
            # Boundary & Obstacle Check
            if self.gt_map_normalized[iy, ix] < self.cfg.collision_threshold:
                reward += self.cfg.collision_penalty
                # Agent stays at old position on collision
            else:
                self.pos[i] = new_pos[i]
                if self.visited[iy, ix] == 0:
                    reward += self.cfg.reward_visited_cells
                    self.visited[iy, ix] = 1
            
            # Record coordinates and values sensed during this step frame
            val = np.clip(self.gt_map_normalized[iy, ix] + np.random.normal(0, 0.05), 0, 1)
            sended_values_at_step.append((float(self.pos[i][0]), float(self.pos[i][1]), float(val)))
        
        self.steps += 1
        
        # Crucial info packet containing path arrays for Streamlit rendering loops
        info = {
            "visited": self.visited,
            "sensed_values": sended_values_at_step
        }
        
        return self._get_obs(), reward, False, self.steps >= self.cfg.max_steps, info