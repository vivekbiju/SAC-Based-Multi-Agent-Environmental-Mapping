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
        self.pos = np.array([self._find_valid_pos() for _ in range(self.num_agents)])
        self.visited = np.zeros(self.map_size)
        self.steps = 0
        return self._get_obs(), {}

    def _find_valid_pos(self):
        while True:
            y, x = np.random.randint(0, self.map_size[0]), np.random.randint(0, self.map_size[1])
            if self.gt_map_normalized[y, x] >= self.cfg.collision_threshold:
                return [x, y]

    def _get_obs(self):
        obs = []
        for p in self.pos:
            val = np.clip(self.gt_map_normalized[int(p[1]), int(p[0])] + np.random.normal(0, 0.05), 0, 1)
            obs.extend([p[0]/self.map_size[1], p[1]/self.map_size[0], val])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        reward = 0
        new_pos = self.pos + action
        for i in range(self.num_agents):
            ix, iy = np.clip(new_pos[i], [0, 0], [self.map_size[1]-1, self.map_size[0]-1]).astype(int)
            if self.gt_map_normalized[iy, ix] < self.cfg.collision_threshold:
                reward += self.cfg.collision_penalty
            else:
                self.pos[i] = new_pos[i]
                if self.visited[iy, ix] == 0:
                    reward += self.cfg.reward_visited_cells
                    self.visited[iy, ix] = 1
        
        self.steps += 1
        return self._get_obs(), reward, False, self.steps >= self.cfg.max_steps, {"visited": self.visited}