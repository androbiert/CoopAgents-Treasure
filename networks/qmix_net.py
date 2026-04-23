import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        
        self.embed_dim = args.mixing_embed_dim
        
        if args.hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(args.state_shape, self.embed_dim * args.n_agents)
            self.hyper_w_final = nn.Linear(args.state_shape, self.embed_dim)
        elif args.hypernet_layers == 2:
            hypernet_embed = args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(args.state_shape, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * args.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(args.state_shape, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        
        # V(s) instead of a bias for each agent
        self.hyper_b_1 = nn.Linear(args.state_shape, self.embed_dim)
        
        # V(s) for final output
        self.V = nn.Sequential(nn.Linear(args.state_shape, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        """
        agent_qs: [batch_size, n_agents]
        states: [batch_size, state_shape]
        """
        bs = agent_qs.size(0)
        
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        w1 = w1.view(bs, self.args.n_agents, self.embed_dim) # [bs, n_agents, embed_dim]
        
        b1 = self.hyper_b_1(states)
        b1 = b1.view(bs, 1, self.embed_dim) # [bs, 1, embed_dim]
        
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(bs, self.embed_dim, 1) # [bs, embed_dim, 1]
        
        v = self.V(states).view(bs, 1, 1) # [bs, 1, 1]
        
        # Add a dimension to agent_qs: [bs, 1, n_agents]
        agent_qs = agent_qs.view(bs, 1, self.args.n_agents)
        
        # Forward pass
        # [bs, 1, n_agents] * [bs, n_agents, embed_dim] -> [bs, 1, embed_dim]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # [bs, 1, embed_dim] * [bs, embed_dim, 1] -> [bs, 1, 1]
        y = torch.bmm(hidden, w_final) + v
        
        q_tot = y.view(bs, -1)
        return q_tot
