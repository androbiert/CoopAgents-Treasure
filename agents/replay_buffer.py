import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, max_ep_len, args):
        self.capacity = capacity
        self.max_ep_len = max_ep_len
        self.args = args
        self.buffer = []
        self.position = 0

    def push(self, episode):
        # episode is a dict of lists
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        
        # Batch needs to be arrays of shape [batch_size, max_ep_len, ...]
        # We assume episodes are padded to max_ep_len or have exactly max_ep_len (for simplicity)
        # Or we can pad them here.
        # Format of batch elements:
        # states: [bs, max_ep_len+1, state_shape] 
        # obs: [bs, max_ep_len+1, n_agents, obs_shape]
        # actions: [bs, max_ep_len, n_agents, 1]
        # rewards: [bs, max_ep_len, 1]
        # dones: [bs, max_ep_len, 1]
        
        keys = ["states", "obs", "actions", "rewards", "dones"]
        res = {}
        for key in keys:
            res[key] = torch.tensor(np.array([ep[key] for ep in batch]), dtype=torch.float32)
        
        res["actions"] = res["actions"].to(torch.int64)
        return res

    def __len__(self):
        return len(self.buffer)
