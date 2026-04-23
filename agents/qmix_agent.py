import torch
import torch.nn as nn
import torch.optim as optim
from networks.agent_net import RNNAgent
from networks.qmix_net import QMixNet
import numpy as np

class QMixAgent:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        
        self.eval_agent = RNNAgent(args.obs_shape, args)
        self.target_agent = RNNAgent(args.obs_shape, args)
        
        self.eval_mix = QMixNet(args)
        self.target_mix = QMixNet(args)
        
        self.target_agent.load_state_dict(self.eval_agent.state_dict())
        self.target_mix.load_state_dict(self.eval_mix.state_dict())
        
        self.eval_parameters = list(self.eval_agent.parameters()) + list(self.eval_mix.parameters())
        self.optimizer = optim.Adam(self.eval_parameters, lr=args.lr)
        
        self.criterion = nn.MSELoss()
        
    def init_hidden(self):
        return self.eval_agent.init_hidden()
        
    def select_action(self, obs, hidden_state, epsilon):
        # obs: [n_agents, obs_shape]
        obs_tensor = torch.FloatTensor(np.array(obs))
        hidden_state = hidden_state.view(self.n_agents, -1)
        
        q_vals, hidden_state = self.eval_agent(obs_tensor, hidden_state)
        
        actions = []
        for i in range(self.n_agents):
            if np.random.rand() < epsilon:
                act = np.random.randint(self.n_actions)
            else:
                act = torch.argmax(q_vals[i]).item()
            actions.append(act)
            
        return actions, hidden_state

    def train(self, batch, max_ep_len):
        states = batch["states"]     # [bs, max_ep_len+1, state_shape]
        obs = batch["obs"]           # [bs, max_ep_len+1, n_agents, obs_shape]
        actions = batch["actions"]   # [bs, max_ep_len, n_agents, 1]
        rewards = batch["rewards"]   # [bs, max_ep_len, 1]
        dones = batch["dones"]       # [bs, max_ep_len, 1]
        # avail_actions = batch["avail_actions"] # Unused for simplicity, assuming all 5 valid
        
        bs = states.size(0)
        
        # Calculate Q_vals for all timesteps
        mac_out = []
        hidden = self.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
        for t in range(max_ep_len + 1):
            q, hidden = self.eval_agent(obs[:, t].reshape(-1, self.args.obs_shape), hidden)
            q = q.reshape(bs, self.n_agents, -1)
            mac_out.append(q)
        mac_out = torch.stack(mac_out, dim=1) # [bs, max_ep_len+1, n_agents, n_actions]
        
        # Calculate target Q_vals
        target_mac_out = []
        target_hidden = self.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
        with torch.no_grad():
            for t in range(max_ep_len + 1):
                target_q, target_hidden = self.target_agent(obs[:, t].reshape(-1, self.args.obs_shape), target_hidden)
                target_q = target_q.reshape(bs, self.n_agents, -1)
                target_mac_out.append(target_q)
        target_mac_out = torch.stack(target_mac_out, dim=1)
        
        # Pick Q-values for actions taken
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) # [bs, max_ep_len, n_agents]
        
        # Double Q-learning step: argmax over eval_mac, eval over target_mac
        max_action_qvals = mac_out[:, 1:].max(dim=3, keepdim=True)[1]
        target_max_qvals = torch.gather(target_mac_out[:, 1:], dim=3, index=max_action_qvals).squeeze(3)
        
        # Mix
        q_tot = self.eval_mix(chosen_action_qvals.reshape(-1, self.n_agents), states[:, :-1].reshape(-1, self.args.state_shape))
        q_tot = q_tot.view(bs, max_ep_len, 1)
        
        target_q_tot = self.target_mix(target_max_qvals.reshape(-1, self.n_agents), states[:, 1:].reshape(-1, self.args.state_shape))
        target_q_tot = target_q_tot.view(bs, max_ep_len, 1)
        
        # Targets
        targets = rewards + self.args.gamma * (1 - dones) * target_q_tot
        
        # TD loss
        loss = self.criterion(q_tot, targets.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        
        return loss.item()
        
    def update_targets(self):
        self.target_agent.load_state_dict(self.eval_agent.state_dict())
        self.target_mix.load_state_dict(self.eval_mix.state_dict())
