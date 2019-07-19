import collections

import numpy as np
import tensorflow as tf

OBSERVATION_KEY = 'observation'
ACHIEVED_GOAL_KEY = 'achieved_goal'
DESIRED_GOAL_KEY = 'desired_goal'


def concat_state_goal(states, goals):
    return np.concatenate((states, goals), axis=-1)


def get_target_updates(vars, target_vars, tau):
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)

class Replay_Memory():

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.states = None
        self.actions = None 
        self.rewards = None
        self.nexts = None
        self.are_non_terminal = None
        self.cursor = -1
        self.is_full = False
    
    def sample_batch(self, batch_size):
        assert self.cursor >= 0
        if self.is_full:
            high = self.memory_size
        else:
            high = self.cursor
        indices = np.random.randint(low=0, high=high, size=batch_size)
        return self.states[indices], \
               self.actions[indices], \
               self.rewards[indices], \
               self.nexts[indices], \
               self.are_non_terminal[indices]
    
    def append(self, states, actions, rewards, nexts, are_non_terminal):
        # Appends transition to the memory.
        n_sample = len(states)
        assert len(states) == len(actions) and \
               len(actions) == len(rewards) and \
               len(rewards) == len(nexts) and \
               len(nexts) == len(are_non_terminal)
        
        if self.cursor == -1:
            state_shape = len(states[0])
            action_shape = len(actions[0])
            self.states = np.zeros((self.memory_size, state_shape))
            self.actions = np.zeros((self.memory_size, action_shape))
            if len(rewards.shape) == 1:
                self.rewards = np.zeros(self.memory_size)
            else:
                self.rewards = np.zeros((self.memory_size, len(rewards[0])))
            self.nexts = np.zeros((self.memory_size, state_shape))
            if len(are_non_terminal.shape) == 1:
                self.are_non_terminal = np.zeros(self.memory_size)
            else:
                self.are_non_terminal = np.zeros((self.memory_size, len(are_non_terminal[0])))
            self.cursor = 0
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        nexts = np.array(nexts)
        are_non_terminal = np.array(are_non_terminal)
        if self.cursor+n_sample >= self.memory_size:
            amount = self.memory_size - self.cursor
            if not self.is_full:
                self.is_full = True
        else:
            amount = n_sample
        self.states[self.cursor:self.cursor+amount] = states[:amount]
        self.actions[self.cursor:self.cursor+amount] = actions[:amount]
        self.rewards[self.cursor:self.cursor+amount] = rewards[:amount]
        self.nexts[self.cursor:self.cursor+amount] = nexts[:amount]
        self.are_non_terminal[self.cursor:self.cursor+amount] = are_non_terminal[:amount]
        self.cursor = (self.cursor+amount)%self.memory_size
        if amount < n_sample:
            remain = n_sample - amount
            self.states[self.cursor:self.cursor+remain] = states[amount:]
            self.actions[self.cursor:self.cursor+remain] = actions[amount:]
            self.rewards[self.cursor:self.cursor+remain] = rewards[amount:]
            self.nexts[self.cursor:self.cursor+remain] = nexts[amount:]
            self.are_non_terminal[self.cursor:self.cursor+remain] = are_non_terminal[amount:]
            self.cursor = (self.cursor+remain)%self.memory_size

class Actor(object):
    def __init__(self, hidden_dims, action_dim, scope):
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.scope = scope

    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.variance_scaling())
            return tf.layers.dense(hidden, self.action_dim, activation=tf.nn.tanh,
                                   kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3))


class Critic(object):
    def __init__(self, hidden_dims, scope):
        self.hidden_dims = hidden_dims
        self.scope = scope

    def __call__(self, states, actions):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # follow the practice of Open AI baselines
            hidden = tf.concat([states, actions], 1)
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.variance_scaling())
            return tf.layers.dense(hidden, 1, activation=None,
                                   kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3))

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)
    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape