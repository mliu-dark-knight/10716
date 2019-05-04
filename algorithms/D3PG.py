from algorithms.DDPG import *

class DistCritic(object):
	
	def __init__(self, action_dim, hidden_dims, n_atom, scope):
		self.action_dim = action_dim
		self.hidden_dims = hidden_dims
		self.n_atom = n_atom
		self.scope = scope
	def __call__(self, states, actions):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			# follow the practice of Open AI baselines
			hidden = tf.concat([states, actions], axis=1)
			for hidden_dim in self.hidden_dims:
				hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
				                         kernel_initializer=tf.initializers.variance_scaling())
			return tf.layers.dense(hidden, self.n_atom, activation=tf.nn.softmax,
			                       kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3))


class D3PG(DDPG):

	def __init__(self, *args, n_atom=51, v_min=-50.0, v_max=0., batch_size=256, **kwargs):
		self.n_atom = n_atom
		self.v_min = float(v_min) 
		self.v_max = float(v_max)
		self.dz = (v_max - v_min) / (self.n_atom - 1.)
		self.atoms = tf.linspace(self.v_min, self.v_max, n_atom)
		self.batch_size = batch_size
		super(D3PG, self).__init__(*args, **kwargs)

	def build_actor_critic(self):
		self.actor = Actor(self.hidden_dims, self.action_dim, 'actor')
		self.actor_target = Actor(self.hidden_dims, self.action_dim, 'target_actor')
		self.critic = DistCritic(self.action_dim, self.hidden_dims, self.n_atom, 'critic')
		self.critic_target = DistCritic(self.action_dim, self.hidden_dims, self.n_atom, 'target_critic')

	def build_loss(self):
		with tf.variable_scope('normalize_states'):
			bn = tf.layers.BatchNormalization(_reuse=tf.AUTO_REUSE)
			states = bn.apply(self.states, training=self.training)
			nexts = bn.apply(self.nexts, training=self.training)
		target_Z = tf.expand_dims(self.rewards, axis=1) + tf.expand_dims(self.are_non_terminal, axis=1) * \
		    np.power(self.gamma, self.N) * self.critic_target(nexts, self.actor_target(nexts))
		# project target_Z back to atoms
		target_Z = tf.minimum(self.v_max, tf.maximum(self.v_min, target_Z))
		b = (target_Z - self.v_min) / self.dz
		l = tf.cast(tf.floor(b), tf.uint32)
		u = tf.cast(tf.ceil(b), tf.uint32)
		print(u.shape)
		print(nexts)
		exit()

		self.critic_loss = -tf.reduce_sum(m*tf.log(self.critic(states, self.actions)))
		self.policy = self.actor(states)
		self.actor_loss = -tf.reduce_mean(tf.reduce_mean(self.critic(states, self.policy), axis=1), axis=0)
