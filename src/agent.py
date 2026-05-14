import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
from src.models import QNetwork, GaussianPolicy
import torch.nn as nn

class SACAgent:
    def __init__(self, state_dim, action_dim, args):
        self.args = args
        self.device = args.device
        
        self.policy = GaussianPolicy(state_dim, action_dim, args.hidden_size).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim, args.hidden_size).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, args.hidden_size).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, args.hidden_size).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, args.hidden_size).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.p_opt = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=args.lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=args.lr)
        
        self.memory = deque(maxlen=args.replay_size)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.policy(state)
            if eval: return torch.tanh(mu).cpu().numpy()[0]
            action, _ = self.policy.sample(state)
        return action.cpu().numpy()[0]

    def update(self):
        if len(self.memory) < self.args.batch_size: return
        batch = random.sample(self.memory, self.args.batch_size)
        s, a, r, s_, d = map(lambda x: torch.FloatTensor(np.array(x)).to(self.device), zip(*batch))
        r, d = r.unsqueeze(1), d.unsqueeze(1)

        with torch.no_grad():
            next_a, next_lp = self.policy.sample(s_)
            q_t = r + (1-d) * self.args.gamma * (torch.min(self.q1_target(s_, next_a), self.q2_target(s_, next_a)) - 0.2 * next_lp)

        # Q Updates
        self.q1_opt.zero_grad(); nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        torch.nn.functional.mse_loss(self.q1(s, a), q_t).backward(); self.q1_opt.step()
        
        self.q2_opt.zero_grad()
        torch.nn.functional.mse_loss(self.q2(s, a), q_t).backward(); self.q2_opt.step()

        # Policy Update
        new_a, lp = self.policy.sample(s)
        self.p_opt.zero_grad()
        (0.2 * lp - torch.min(self.q1(s, new_a), self.q2(s, new_a))).mean().backward(); self.p_opt.step()

        # Soft Update
        for t, s in zip([self.q1_target, self.q2_target], [self.q1, self.q2]):
            for tp, p in zip(t.parameters(), s.parameters()):
                tp.data.copy_(tp.data * (1 - self.args.tau) + p.data * self.args.tau)