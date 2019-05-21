from functools import reduce
from operator import mul
from typing import *

from tqdm import tqdm

from algorithms.common import *
from algorithms.QRDDPG import QRCritic
from utils import append_summary
import os


class QRQNetwork(object):
	def __init__(self, action_dim, hidden_dims, n_quantile, scope):
		self.action_dim = action_dim
		self.hidden_dims = hidden_dims
		self.n_quantile = n_quantile
		self.scope = scope
	def __call__(self, states):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			hidden = states
			for hidden_dim in self.hidden_dims:
				hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
				                         kernel_initializer=tf.initializers.variance_scaling())
			hidden = tf.layers.dense(hidden, self.n_quantile*self.action_dim, activation=None,
			                       kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3))
			return tf.reshape(hidden, [-1, self.action_dim, self.n_quantile])


class QRDQN(object):

	def __init__(self, env, hidden_dims=[256, 256], tau=1e-2, kappa=1.0, n_quantile=64, replay_memory=None, gamma=1.0,
	             lr=1e-3, N=5, scope_pre = ""):
		self.env = env
		self.hidden_dims = hidden_dims
		self.state_dim = reduce(mul,env.observation_space.shape)
		self.n_action = env.action_space.n
		self.kappa = kappa
		self.n_quantile = n_quantile
		self.gamma = gamma
		self.lr = lr
		self.N = N
		self.replay_memory = replay_memory
		self.tau = tau
		self.scope_pre = scope_pre
		self.build()

	def build(self):
		self.build_network()
		self.build_placeholder()
		self.build_loss()
		self.build_copy_op()
		self.build_step()
		self.build_summary()

	def build_network(self):
		self.qnetwork = QRQNetwork(self.n_action, self.hidden_dims, self.n_quantile, self.scope_pre+'qnetwork')
		self.qnetwork_target = QRQNetwork(self.n_action, self.hidden_dims, self.n_quantile, self.scope_pre+'target_qnetwork')

	def build_placeholder(self):
		self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		self.actions = tf.placeholder(tf.int32, shape=[None,1])
		self.rewards = tf.placeholder(tf.float32, shape=[None])
		self.nexts = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		self.are_non_terminal = tf.placeholder(tf.float32, shape=[None])
		self.training = tf.placeholder(tf.bool)

	def build_copy_op(self):
		self.init_qnetwork, self.update_qnetwork = get_target_updates(
			tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_pre+'qnetwork'),
			tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_pre+'target_qnetwork'), self.tau)

	def build_loss(self):
		with tf.variable_scope(self.scope_pre+'normalize_states'):
			bn = tf.layers.BatchNormalization(_reuse=tf.AUTO_REUSE)
			states = bn.apply(self.states, training=self.training)
			nexts = bn.apply(self.nexts, training=self.training)
		batch_size = tf.shape(states)[0]
		batch_indices = tf.range(batch_size,dtype=tf.int32)[:, None]
		
		target_network_output = self.qnetwork_target(nexts)
		target_qvalue = tf.reduce_mean(target_network_output, axis=2)
		target_action = tf.cast(tf.argmax(target_qvalue, axis=1)[:, None],
							    dtype=tf.int32)
		target_action_indices = tf.concat([batch_indices, target_action], axis=1)
		target_quantiles = tf.gather_nd(target_network_output,
										target_action_indices)
		print_op = tf.print(self.rewards)
		#with tf.control_dependencies([print_op]):
		target_Z = tf.expand_dims(self.rewards, axis=1) + tf.expand_dims(self.are_non_terminal, axis=1) * \
		           np.power(self.gamma, self.N) * target_quantiles
		
		self.network_output = self.qnetwork(states) 
		self.qvalue = tf.reduce_mean(self.network_output, axis=2) 
		action_indices = tf.concat([batch_indices, self.actions], axis=1)
		Z = tf.gather_nd(self.network_output, action_indices)
		bellman_errors = tf.expand_dims(target_Z, axis=2) - tf.expand_dims(Z, axis=1)
		huber_loss_case_one = tf.to_float(tf.abs(bellman_errors) <= self.kappa) * 0.5 * bellman_errors ** 2
		huber_loss_case_two = tf.to_float(tf.abs(bellman_errors) > self.kappa) * self.kappa * \
		                      (tf.abs(bellman_errors) - 0.5 * self.kappa)
		huber_loss = huber_loss_case_one + huber_loss_case_two
		quantiles = tf.expand_dims(tf.expand_dims(tf.range(0.5 / self.n_quantile, 1., 1. / self.n_quantile), axis=0),
		                           axis=1)
		quantile_huber_loss = (tf.abs(quantiles - tf.stop_gradient(tf.to_float(bellman_errors < 0))) * huber_loss) / \
		                      self.kappa

		# Sum over current quantile value (num_tau_samples) dimension,
		# average over target quantile value (num_tau_prime_samples) dimension.
		# Shape: batch_size x num_tau_prime_samples x 1.
		self.loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=2), axis=1), axis=0)

	def build_step(self):
		self.global_step = tf.Variable(0, trainable=False)
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		with tf.control_dependencies([tf.Assert(tf.is_finite(self.loss), [self.loss])]):
			self.step = optimizer.minimize(
				self.loss,
				var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_pre+'qnetwork'))

	def build_summary(self):
		tf.summary.scalar('loss', self.loss)
		self.merged_summary_op = tf.summary.merge_all()

	def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
	          train_episodes=1000, save_episodes=100, epsilon=0.3):
		total_rewards = []
		sess.run([self.init_qnetwork])
		for i_episode in tqdm(range(train_episodes), ncols=100):
			cur_epsilon = self.linear_decay_epsilon(i_episode, train_episodes*0.5, epsilon)
			total_reward = self.collect_trajectory(cur_epsilon)
			append_summary(progress_fd, str(start_episode + i_episode) + ',{0:.2f}'.format(total_reward))
			total_rewards.append(total_reward)
			states, actions, rewards, nexts, are_non_terminal = self.replay_memory.sample_batch(step * batch_size)
			for t in range(step):
				sess.run([self.step],
				         feed_dict={self.states: states[t * batch_size: (t + 1) * batch_size],
				                    self.actions: actions[t * batch_size: (t + 1) * batch_size],
				                    self.rewards: rewards[t * batch_size: (t + 1) * batch_size],
				                    self.nexts: nexts[t * batch_size: (t + 1) * batch_size],
				                    self.are_non_terminal: are_non_terminal[t * batch_size: (t + 1) * batch_size],
				                    self.training: True})
				sess.run([self.update_qnetwork])
			# summary_writer.add_summary(summary, global_step=self.global_step.eval())
			if (i_episode + 1) % save_episodes == 0:
				saver.save(sess, model_path)
		return total_rewards

	def linear_decay_epsilon(self, step, n_step, epsilon):
		if step <n_step:
			return np.linspace(epsilon, 1, n_step)[-step]
		else:
			return epsilon


	def collect_trajectory(self, epsilon):
		states, actions, rewards = self.generate_episode(epsilon=epsilon)
		states = np.array(states)
		returns, nexts, are_non_terminal = self.normalize_returns(states[1:], rewards)
		total_reward = sum(rewards)
		self.replay_memory.append(states[:-1],
		                          actions, returns,
		                          nexts,
		                          are_non_terminal)

		return total_reward

	def normalize_returns(self, nexts: np.array, rewards: List) -> Tuple[List, np.array, List]:
		'''
		Compute N step returns
		:param states:
		:param rewards:
		:return:
		WARNING: make sure states, rewards come from the same episode
		'''
		T = len(nexts)
		assert len(rewards) == T
		nexts = np.roll(nexts, -self.N, axis=0)
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

	def generate_episode(self, epsilon=0.0, render=False):
		'''
		:param epsilon: exploration noise
		:return:
		WARNING: make sure whatever you return comes from the same episode
		'''
		states = []
		actions = []
		rewards = []

		state = self.env.reset()

		while True:
			states.append(state)
			qvalue = self.qvalue.eval(feed_dict={
				self.states: np.expand_dims(state, axis=0),
				self.training: False}).squeeze(axis=0)
			action = np.argmax(qvalue)
			#print("qvalues: {}".format(qvalue))
			#print("action: {}".format(action))
			#print("epsilon:{}".format(epsilon))
			if np.random.random() < epsilon:
				action = np.random.randint(low=0, high=self.n_action)
			state, reward, done, _ = self.env.step(action)
			if render:
				self.env.render()
			actions.append([action])
			rewards.append(reward)
			if done:
				break
		states.append(state)
		assert len(states)  == len(actions)+1 and \
		       len(actions) == len(rewards)
		#print("epi reward: {}".format(rewards))
		return states, actions, rewards

class EnvironmentWrapper(object):
	def __init__(self, observation_space, action_spcae):
		self.observation_space = observation_space
		self.action_space = action_spcae

class MultiagentWrapper(object):
	def __init__(self, environment, n_agent, agent_params):
		## only consider QRDQN now
		self.env = environment
		self.n_agent = n_agent
		self.agents = []
		for i in range(self.n_agent):
			obs = self.env.observation_space[i]
			act = self.env.action_space[i]
			env = EnvironmentWrapper(obs, act)
			agent = QRDQN(env=env, **agent_params[i])
			self.agents.append(agent)
	
	def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
	          train_episodes=1000, save_episodes=100, epsilon=0.3, max_episode_len=25):
		total_rewards = []
		sess.run([agent.init_qnetwork for agent in self.agents])
		for i_episode in tqdm(range(train_episodes), ncols=100):
			cur_epsilon = self.linear_decay_epsilon(i_episode, train_episodes*0.5, epsilon)
			total_reward = self.collect_trajectory(cur_epsilon, max_episode_len)
			append_summary(progress_fd, str(start_episode + i_episode) + ',{0:.2f}'.format(total_reward))
			total_rewards.append(total_reward)
			for agent in self.agents:
				states, actions, rewards, nexts, are_non_terminal = agent.replay_memory.sample_batch(step * batch_size)
				for t in range(step):
					sess.run([agent.step],
				        feed_dict={agent.states: states[t * batch_size: (t + 1) * batch_size],
				                    agent.actions: actions[t * batch_size: (t + 1) * batch_size],
				                    agent.rewards: rewards[t * batch_size: (t + 1) * batch_size],
				                    agent.nexts: nexts[t * batch_size: (t + 1) * batch_size],
				                    agent.are_non_terminal: are_non_terminal[t * batch_size: (t + 1) * batch_size],
				                    agent.training: True})
				sess.run([agent.update_qnetwork])
			# summary_writer.add_summary(summary, global_step=self.global_step.eval())
			if (i_episode + 1) % save_episodes == 0:
				saver.save(sess, model_path)
		return total_rewards

	def linear_decay_epsilon(self, step, n_step, epsilon):
		if step <n_step:
			return np.linspace(epsilon, 1, n_step)[-step]
		else:
			return epsilon


	def collect_trajectory(self, epsilon, max_episode_len):
		states, actions, rewards = self.generate_episode(epsilon=epsilon, max_episode_len=max_episode_len)
		actions = np.array(actions)
		rewards = np.array(rewards)
		for agent_id in range(self.n_agent):
			agent = self.agents[agent_id]
			agent_states = np.array(states[agent_id])
			returns, nexts, are_non_terminal = agent.normalize_returns(agent_states[1:], rewards[:, agent_id])
			agent.replay_memory.append(agent_states[:-1],
		                          actions[:, agent_id], returns,
		                          nexts,
		                          are_non_terminal)

		return rewards[:, 0].sum()

	def generate_episode(self, epsilon=0.0, render=False,
						 max_episode_len=25, benchmark=False):
		'''
		:param epsilon: exploration noise
		:return:
		WARNING: make sure whatever you return comes from the same episode
		'''
		states = []
		actions = []
		rewards = []
		state = self.env.reset()
		for i in range(self.n_agent):
			states.append([])
		step = 0
		while True:
			action = []
			for i in range(self.n_agent):
				states[i].append(state[i])
			for agent_id in range(self.n_agent):
				agent = self.agents[agent_id]
				qvalue = self.agents[agent_id].qvalue.eval(feed_dict={
					agent.states: np.expand_dims(state[agent_id], axis=0),
					agent.training: False}).squeeze(axis=0)
				agent_action = np.argmax(qvalue)
				if np.random.random() < epsilon:
					agent_action = np.random.randint(low=0, high=agent.n_action)
				action.append(agent_action)
			state, reward, done, info = self.env.step(action)
			step += 1
			if render:
				self.env.render()
			action = np.array(action).reshape((len(action),1))
			actions.append(action)
			rewards.append(reward)
			if all(done) or step >= max_episode_len:
				#print("Break. Step={}".format(step))
				break
		for i in range(self.n_agent):
				states[i].append(state[i])
		assert len(states[0])  == len(actions)+1 and \
		       len(actions) == len(rewards)
		if not benchmark:
			return states, actions, rewards
		else:
			return states, actions, rewards, info