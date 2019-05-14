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
			self.rewards = np.zeros(self.memory_size)
			self.nexts = np.zeros((self.memory_size, state_shape))
			self.are_non_terminal = np.zeros(self.memory_size)
			self.cursor = 0
		
		states = np.array(states)
		actions = np.array(actions)
		rewards = np.array(rewards)
		nexts = np.array(nexts)
		are_non_terminal = np.array(are_non_terminal)
		if self.cursor+n_sample > self.memory_size:
			amount = self.memory_size - self.cursor
			if not self.is_full:
				self.is_full = True
		else:
			amount = n_sample
		#print("amount: {}".format(amount))
		#print("cursor: {}".format(self.cursor))
		#print("n_sample: {}".format(n_sample))
		self.states[self.cursor:self.cursor+amount] = states[:amount]
		self.actions[self.cursor:self.cursor+amount] = actions[:amount]
		self.rewards[self.cursor:self.cursor+amount] = rewards[:amount]
		self.nexts[self.cursor:self.cursor+amount] = nexts[:amount]
		self.are_non_terminal[self.cursor:self.cursor+amount] = are_non_terminal[:amount]
		self.cursor = self.cursor+amount
		if self.cursor >= self.memory_size:
			self.cursor = 0
		if amount < n_sample:
			remain = n_sample - amount
			self.states[self.cursor:self.cursor+remain] = states[amount:]
			self.actions[self.cursor:self.cursor+remain] = actions[amount:]
			self.rewards[self.cursor:self.cursor+remain] = rewards[amount:]
			self.nexts[self.cursor:self.cursor+remain] = nexts[amount:]
			self.are_non_terminal[self.cursor:self.cursor+remain] = are_non_terminal[amount:]
			self.cursor = (self.cursor+remain)%self.memory_size

class Replay_Memory_():

	def __init__(self, memory_size):
		self.states = collections.deque([], maxlen=memory_size)
		self.actions = collections.deque([], maxlen=memory_size)
		# cumulative reward throughout N steps
		self.rewards = collections.deque([], maxlen=memory_size)
		# next state after N steps
		self.nexts = collections.deque([], maxlen=memory_size)
		# next state is terminal or not
		self.are_non_terminal = collections.deque([], maxlen=memory_size)

	def sample_batch(self, batch_size):
		indices = np.random.choice(len(self.states), 
									   size=batch_size, replace=True)
		return np.array(self.states)[indices], \
		       np.array(self.actions)[indices], \
		       np.array(self.rewards)[indices], \
		       np.array(self.nexts)[indices], \
		       np.array(self.are_non_terminal)[indices]

	def append(self, states, actions, rewards, nexts, are_non_terminal):
		# Appends transition to the memory.
		assert len(states) == len(actions) and \
		       len(actions) == len(rewards) and \
		       len(rewards) == len(nexts) and \
		       len(nexts) == len(are_non_terminal)
		self.states += states
		self.actions += actions
		self.rewards += rewards
		self.nexts += nexts
		self.are_non_terminal += are_non_terminal

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
