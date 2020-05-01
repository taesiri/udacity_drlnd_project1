import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)   # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.999            # discount factor
TAU = 1e-3               # for soft update of target parameters
LR = 5e-4                # learning rate 
UPDATE_EVERY = 4         # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QAgentGeneric():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, local_brain, target_brain, update_rule='dqn', seed=0):
        """ Generic DQN Agent - Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            update_rule (string): updating rule of the network, `dqn` or `double_dqn`
            brain_size (list): An int list, containing size of hidden layers of the agent's neural network
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local  = local_brain   #
        self.qnetwork_target = target_brain #
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Update Rule
        self.update_rule = update_rule
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval() # Set to Evaluation Mode
        with torch.no_grad(): # Dont Store Gradients
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # Set Back to Train Mode

        # Epsilon-greedy action selection 
        # We calculate action_values, but don't use them? where else we need action values for this step?
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        state_batch, action_batch, reward_batch, next_states_batch, doneMask = experiences
        ## Choose value corresponds to each action in the result
        state_action_values = self.qnetwork_local(state_batch).gather(1, action_batch).squeeze() 

        expected_state_action_values = None

        if self.update_rule == "dqn":
            next_state_values = self.qnetwork_target(next_states_batch).max(1)[0].detach() ## detach --> we don't want to backbropagate through Target NETWORK!
            expected_state_action_values = torch.mul((gamma*next_state_values), doneMask.squeeze()) + reward_batch.squeeze()

        elif self.update_rule == "double_dqn":
            max_values, max_indices = self.qnetwork_local(next_states_batch).detach().max(1)
            estimated_value = self.qnetwork_target(next_states_batch).gather(1, max_indices.view(-1,1)).view(-1).detach()
            expected_state_action_values = torch.mul((gamma * estimated_value), doneMask.squeeze()) + reward_batch.squeeze()
        else:
            raise Exception(f'Update rule is not implemented {self.update_rule}')


        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)               # Gradient cliping
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([1-e.done for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)