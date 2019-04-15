from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from algorithms.common import Actor, Critic, Replay_Memory, get_target_updates


class DDPG(object):

	def __init__(self, env, hidden_dims, gamma=1.0, init_std=0.2, tau=1e-2,
				 actor_lr=1e-3, critic_lr=1e-3, N=5, memory_size=10000, delta=1.0):
		self.env = env
		self.state_dim = reduce(mul, env.observation_space.spaces['observation'].shape)
		self.action_dim = reduce(mul, self.env.action_space.shape)

		self.gamma = gamma
		self.init_std = init_std
		self.tau = tau
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		self.N = N
		self.delta = delta

		self.actor = Actor(hidden_dims, self.action_dim, 'actor')
		self.actor_target = Actor(hidden_dims, self.action_dim, 'target_actor')
		self.critic = Critic(hidden_dims, 'critic')
		self.critic_target = Critic(hidden_dims, 'target_critic')
		self.replay_memory = Replay_Memory(memory_size=memory_size)
		self.build()

	def build(self):
		self.build_placeholder()
		self.build_loss()
		self.build_copy_op()
		self.build_step()

	def build_placeholder(self):
		self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		self.actions = tf.placeholder(tf.float32, shape=[None, self.action_dim])
		self.rewards = tf.placeholder(tf.float32, shape=[None])
		self.nexts = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		self.are_non_terminal = tf.placeholder(tf.float32, shape=[None])
		self.training = tf.placeholder(tf.bool)

	def build_copy_op(self):
		self.init_actor, self.update_actor = get_target_updates(
			tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'),
			tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor'), self.tau)

		self.init_critic, self.update_critic = get_target_updates(
			tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'),
			tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic'), self.tau)

	def build_loss(self):
		with tf.variable_scope('normalize_states'):
			bn = tf.layers.BatchNormalization(_reuse=tf.AUTO_REUSE)
			states = bn.apply(self.states, training=self.training)
			nexts = bn.apply(self.nexts, training=self.training)

		Q_target = self.rewards + self.are_non_terminal * np.power(self.gamma, self.N) * \
				   tf.squeeze(self.critic_target(nexts, self.actor_target(nexts, self.training), self.training), axis=1)
		Q = tf.squeeze(self.critic(states, self.actions, self.training), axis=1)
		self.critic_loss = tf.losses.mean_squared_error(Q_target, Q)
		self.policy = self.actor(states, self.training)
		self.actor_loss = -tf.reduce_mean(self.critic(states, self.policy, self.training))

	def build_step(self):
		actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
		with tf.control_dependencies([tf.Assert(tf.is_finite(self.actor_loss), [self.actor_loss])]):
			self.actor_step = actor_optimizer.minimize(
				self.actor_loss,
				var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'))
		critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='normalize_states') +
									 [tf.Assert(tf.is_finite(self.critic_loss), [self.critic_loss])]):
			self.critic_step = critic_optimizer.minimize(
				self.critic_loss,
				var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))

	def train(self, sess, saver, model_path, batch_size=64, step=10, train_episodes=1000, save_episodes=100):
		total_rewards = []
		sess.run([self.init_actor, self.init_critic])
		for i_episode in tqdm(range(train_episodes), ncols=100):
			total_rewards.append(self.collect_trajectory(self.get_noise(i_episode)))
			states, actions, rewards, nexts, are_non_terminal = self.replay_memory.sample_batch(step * batch_size)
			for t in range(step):
				sess.run([self.critic_step],
						 feed_dict={self.states: states[t * batch_size: (t + 1) * batch_size],
									self.actions: actions[t * batch_size: (t + 1) * batch_size],
									self.rewards: rewards[t * batch_size: (t + 1) * batch_size],
									self.nexts: nexts[t * batch_size: (t + 1) * batch_size],
									self.are_non_terminal: are_non_terminal[t * batch_size: (t + 1) * batch_size],
									self.training: True})
				sess.run([self.actor_step],
						 feed_dict={self.states: states[t * batch_size: (t + 1) * batch_size], self.training: True})
				sess.run([self.update_actor, self.update_critic])
			if (i_episode + 1) % save_episodes == 0:
				saver.save(sess, model_path)
		return total_rewards

	def get_noise(self, i_episode):
		return self.init_std / (9e-4 * i_episode + 1)

	def collect_trajectory(self, sigma):
		states, actions, rewards = self.generate_episode(sigma=sigma)
		returns, nexts, are_non_terminal = self.normalize_returns(states, rewards)
		total_reward = sum(rewards)
		self.replay_memory.append(states, actions, list(returns), list(nexts), list(are_non_terminal))

		# use position of last state as new goal
		goal = states[-1][:DDPG.goal_dim]
		states = np.array(states)
		states[:, -DDPG.goal_dim:] = goal
		rewards = -0.1 * (np.linalg.norm(states[:, :DDPG.goal_dim] - goal, axis=1) > self.delta)
		returns, nexts, are_non_terminal = self.normalize_returns(states, rewards)
		self.replay_memory.append(list(states), actions, list(returns), list(nexts), list(are_non_terminal))

		return total_reward

	def normalize_returns(self, states, rewards):
		'''
		:param states:
		:param rewards:
		:return:
		WARNING: make sure states, rewards come from the same episode
		'''
		T = len(states)
		nexts = np.roll(np.array(states), -self.N, axis=0)
		are_non_terminal = np.ones(T)
		if T >= self.N:
			are_non_terminal[T - self.N:] = 0
		else:
			are_non_terminal[:] = 0

		returns = []
		for reward in reversed(rewards):
			return_ = reward if len(returns) == 0 else reward + self.gamma * returns[-1]
			returns.append(return_)
		returns.reverse()
		returns = np.array(returns)
		if T > self.N:
			returns -= np.power(self.gamma, self.N) \
					   * np.pad(returns[self.N:], (0, self.N), 'constant', constant_values=(0))

		return returns, nexts, are_non_terminal

	def generate_episode(self, sigma=0.0):
		'''
		:param sigma: exploration noise
		:return: states, actions, weights of actions, rewards of single episode.
		WARNING: make sure whatever you return comes from the same episode
		'''
		states = []
		actions = []
		rewards = []

		next = self.env.reset()
		while True:
			state = next
			action = self.policy.eval(feed_dict={self.states: np.expand_dims(state, 0), self.training: False}).squeeze()
			action = np.random.normal(loc=action, scale=sigma)
			next, reward, done, _ = self.env.step(action)
			states.append(state)
			actions.append(action)
			rewards.append(reward)
			if done:
				break

		return states, actions, rewards
