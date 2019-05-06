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

    def __init__(self, *args, n_atom=51, v_min=-50.0, v_max=0., **kwargs):
        self.n_atom = n_atom
        self.v_min = float(v_min) 
        self.v_max = float(v_max)
        self.dz = (v_max - v_min) / (self.n_atom - 1.)
        self.atoms = tf.linspace(self.v_min, self.v_max, n_atom)
        super(D3PG, self).__init__(*args, **kwargs)

    def build_actor_critic(self):
        self.actor = Actor(self.hidden_dims, self.action_dim, 'actor')
        self.actor_target = Actor(self.hidden_dims, self.action_dim, 'target_actor')
        self.critic = DistCritic(self.action_dim, self.hidden_dims, self.n_atom, 'critic')
        self.critic_target = DistCritic(self.action_dim, self.hidden_dims, self.n_atom, 'target_critic')

    def build_target_distribution(self):
        with tf.variable_scope('normalize_states'):
            bn = tf.layers.BatchNormalization(_reuse=tf.AUTO_REUSE)
            nexts = bn.apply(self.nexts, training=self.training)
        target_Z = self.critic_target(nexts, self.actor_target(nexts))
        target_atom = tf.expand_dims(self.rewards, axis=1) + tf.expand_dims(self.are_non_terminal, axis=1) * \
            np.power(self.gamma, self.N) * self.atoms
        projected = self.project_distribution(target_atom, target_Z, self.atoms)
        return projected

    def build_loss(self):
        with tf.variable_scope('normalize_states'):
            bn = tf.layers.BatchNormalization(_reuse=tf.AUTO_REUSE)
            states = bn.apply(self.states, training=self.training)
        projected = tf.stop_gradient(self.build_target_distribution())
        probabilities = self.critic(states, self.actions)
        self.critic_loss = -tf.reduce_mean(tf.reduce_sum(projected*tf.log(1e-20+probabilities), axis=1), axis=0)
        self.policy = self.actor(states)
        EZ = tf.reduce_mean(tf.reduce_sum(self.critic(states, self.policy)*tf.expand_dims(self.atoms, axis=0), axis=1), axis=0)
        self.actor_loss = -EZ

    def project_distribution(self, supports, weights, target_support):	
    # see https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/rainbow_agent.py
        target_support_deltas = target_support[1:] - target_support[:-1]
        delta_z = target_support_deltas[0]
        v_min = target_support[0]
        v_max = target_support[-1]
        batch_size = tf.shape(supports)[0]
        num_dims = tf.shape(target_support)[0]
        clipped_support = tf.clip_by_value(supports, v_min, v_max)[:, None, :]
        tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])
        reshaped_target_support = tf.tile(target_support[:, None], [batch_size, 1])
        reshaped_target_support = tf.reshape(reshaped_target_support,
                                         [batch_size, num_dims, 1])
        numerator = tf.abs(tiled_support - reshaped_target_support)
        quotient = 1 - (numerator / delta_z)
        clipped_quotient = tf.clip_by_value(quotient, 0, 1)
        weights = weights[:, None, :]
        inner_prod = clipped_quotient * weights
        projection = tf.reduce_sum(inner_prod, 3)
        projection = tf.reshape(projection, [batch_size, num_dims])
        return projection