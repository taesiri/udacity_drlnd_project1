import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedBrain(nn.Module):
    """Simple Fully Connected Brain"""

    def __init__(self, state_size, action_size, seed, hidden_sizes = [32, 32, 16]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_sizes (list): List of Hidden Layers' size
            seed (int): Random seed
        """
        super(FullyConnectedBrain, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        # A Generic Fully Connection Network.
        layers = zip([state_size] + hidden_sizes, hidden_sizes + [action_size])
        self.fcs = [nn.Linear(h_size[0], h_size[1]) for h_size in layers]
        self.fcs = nn.ModuleList(self.fcs) 

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for layer in self.fcs[:-1]:
            x = F.relu(layer(x))
        return self.fcs[-1](x)


class DuelingFullyConnectedBrain(nn.Module):
    """Dueling Fully Connected Brain for Dueling DQN"""

    def __init__(self, state_size, action_size, seed, shared_hidden = [32, 32, 16], value_head=[16], advantage_head=[16]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            shared_hidden (list): List of Hidden Layers' size for shared part of network
            value_head (list): List of Hidden Layers' size for value_head part of network
            advantage_head (list): List of Hidden Layers' size for advantage_head part of network
            seed (int): Random seed
        """
        super(DuelingFullyConnectedBrain, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        # Shared Backbone
        layers = zip([state_size] + shared_hidden[:-1], shared_hidden)
        self.shared_fcs = [nn.Linear(h_size[0], h_size[1]) for h_size in layers]
        self.shared_fcs = nn.ModuleList(self.shared_fcs)

        # Value Head
        layers = zip([shared_hidden[-1]] + value_head, value_head + [1])
        self.value_fcs = [nn.Linear(vh_size[0], vh_size[1]) for vh_size in layers]
        self.value_fcs = nn.ModuleList(self.value_fcs)

        # Advantage Head
        layers = zip([shared_hidden[-1]] + advantage_head, advantage_head + [action_size])
        self.advantage_fcs = [nn.Linear(ah_size[0], ah_size[1]) for ah_size in layers]
        self.advantage_fcs = nn.ModuleList(self.advantage_fcs)

    def forward(self, state):
        """Build a network that maps state -> Value and Advantage, then map those two to action values."""
        x = state
        for layer in self.shared_fcs:
            x = F.relu(layer(x))

        # VALUE
        v = x
        for layer in self.value_fcs[:-1]:
            v = F.relu(layer(v))
        v = self.value_fcs[-1](v)

        # ADVANTAGE
        a = x
        for layer in self.advantage_fcs[:-1]:
            a = F.relu(layer(a))
        a = self.advantage_fcs[-1](a)

        # Compute Q
        q =  v + a - a.mean()

        return q