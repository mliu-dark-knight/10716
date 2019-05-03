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
		self.actor = Actor(self.hidden_dims, self.action_dim, 'actor')
		self.actor_target = Actor(self.hidden_dims, self.action_dim, 'target_actor')
		self.critic = QRCritic(self.action_dim, self.hidden_dims, self.n_quantile, 'critic')
		self.critic_target = QRCritic(self.action_dim, self.hidden_dims, self.n_quantile, 'target_critic')

	def build_loss(self):
		with tf.variable_scope('normalize_states'):
			bn = tf.layers.BatchNormalization(_reuse=tf.AUTO_REUSE)
			states = bn.apply(self.states, training=self.training)
			nexts = bn.apply(self.nexts, training=self.training)

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
		self.actor_loss = -tf.reduce_mean(tf.reduce_mean(self.critic(states, self.policy), axis=1), axis=0)
