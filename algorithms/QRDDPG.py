from algorithms.DDPG import *


class QRCritic(object):
	def __init__(self, action_dim, hidden_dims, n_quantile, scope):
		self.action_dim = action_dim
		self.hidden_dims = hidden_dims
		self.n_quantile = n_quantile
		self.scope = scope
	def __call__(self, states, actions):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			# follow the practice of Open AI baselines
			hidden = tf.concat([states, actions], axis=1)
			for hidden_dim in self.hidden_dims:
				hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
				                         kernel_initializer=tf.initializers.variance_scaling())
			return tf.layers.dense(hidden, self.n_quantile, activation=None,
			                       kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3))


class QRDDPG(DDPG):

	def __init__(self, *args, kappa=1.0, n_quantile=64, **kwargs):
		self.kappa = kappa
		self.n_quantile = n_quantile
		super(QRDDPG, self).__init__(*args, **kwargs)

	def build_actor_critic(self):
		self.actor = Actor(self.hidden_dims, self.n_action, self.scope_pre+'actor')
		self.actor_target = Actor(self.hidden_dims, self.n_action, self.scope_pre+'target_actor')
		self.critic = QRCritic(self.n_action, self.hidden_dims, self.n_quantile, self.scope_pre+'critic')
		self.critic_target = QRCritic(self.n_action, self.hidden_dims, self.n_quantile, self.scope_pre+'target_critic')

	def build_placeholder(self):
		self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		self.actions = tf.placeholder(tf.float32, shape=[None, self.n_action])
		self.rewards = tf.placeholder(tf.float32, shape=[None])
		self.nexts = tf.placeholder(tf.float32, shape=[None, self.state_dim])
		self.are_non_terminal = tf.placeholder(tf.float32, shape=[None])

	def build_loss(self):
		states = self.states
		nexts = self.nexts

		target_Z = tf.expand_dims(self.rewards, axis=1) + tf.expand_dims(self.are_non_terminal, axis=1) * \
		           np.power(self.gamma, self.N) * self.critic_target(nexts, self.actor_target(nexts))
		Z = self.critic(states, self.actions)

		bellman_errors = tf.expand_dims(target_Z, axis=2) - tf.expand_dims(Z, axis=1)
		# The huber loss (see Section 2.3 of the paper) is defined via two cases:
		# case_one: |bellman_errors| <= kappa
		# case_two: |bellman_errors| > kappa
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
		self.critic_loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=2), axis=1), axis=0)
		self.policy = self.actor(states)
		self.actor_loss = -tf.reduce_mean(self.critic(states, self.policy)[:, int(self.n_quantile/2)], axis=0)

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
			agent = QRDDPG(env=env, **agent_params[i])
			self.agents.append(agent)
	
	def train(self, sess, saver, summary_writer, progress_fd, model_path, filter_path, batch_size=64, step=10, start_episode=0,
	          train_episodes=1000, save_episodes=100, epsilon=0.3, max_episode_len=25):
		total_rewards = []
		sess.run([agent.init_actor for agent in self.agents]+[agent.init_critic for agent in self.agents])
		for i_episode in tqdm(range(train_episodes), ncols=100):
			total_reward = self.collect_trajectory(epsilon, max_episode_len)
			append_summary(progress_fd, str(start_episode + i_episode) + ',{0:.2f}'.format(total_reward))
			total_rewards.append(total_reward)
			for agent in self.agents:
				states, actions, rewards, nexts, are_non_terminal = agent.replay_memory.sample_batch(step * batch_size)
				for t in range(step):
					sess.run([agent.critic_step],
				        feed_dict={agent.states: states[t * batch_size: (t + 1) * batch_size],
				                    agent.actions: actions[t * batch_size: (t + 1) * batch_size],
				                    agent.rewards: rewards[t * batch_size: (t + 1) * batch_size],
				                    agent.nexts: nexts[t * batch_size: (t + 1) * batch_size],
				                    agent.are_non_terminal: are_non_terminal[t * batch_size: (t + 1) * batch_size]})

					sess.run([agent.actor_step],
				        feed_dict={agent.states: states[t * batch_size: (t + 1) * batch_size],
				                    agent.actions: actions[t * batch_size: (t + 1) * batch_size],
				                    agent.rewards: rewards[t * batch_size: (t + 1) * batch_size],
				                    agent.nexts: nexts[t * batch_size: (t + 1) * batch_size],
				                    agent.are_non_terminal: are_non_terminal[t * batch_size: (t + 1) * batch_size]})
				sess.run([agent.update_critic, agent.update_actor])
			# summary_writer.add_summary(summary, global_step=self.global_step.eval())
			if (i_episode + 1) % save_episodes == 0:
				saver.save(sess, model_path)
		return total_rewards

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
				agent_action = agent.policy.eval(feed_dict={
					agent.states: np.expand_dims(state[agent_id], axis=0)}).squeeze(axis=0)
				agent_action = np.random.normal(loc=agent_action, scale=epsilon)
				action.append(agent_action)
			state, reward, done, info = self.env.step(action)
			step += 1
			if render:
				self.env.render()
			action = np.array(action)
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