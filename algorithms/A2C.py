from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from algorithms.common import *
from utils import append_summary

class Actor_(object):
	def __init__(self, hidden_dims, action_dim, scope):
		self.hidden_dims = hidden_dims
		self.action_dim = action_dim
		self.scope = scope

	def __call__(self, states):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			hidden = states
			for hidden_dim in self.hidden_dims:
				hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
				                         kernel_initializer=tf.initializers.variance_scaling(),
				                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
			return tf.layers.dense(hidden, self.action_dim, activation=None,
			                       kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3),
			                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

class BetaActor(object):
    def __init__(self, hidden_dims, action_dim, scope):
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.scope = scope

    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.variance_scaling(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
            alpha = tf.layers.dense(hidden, self.action_dim, activation=tf.nn.softplus,
                                   kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
            beta = tf.layers.dense(hidden, self.action_dim, activation=tf.nn.softplus,
                                   kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
            return alpha+1, beta+1

class VNetwork(object):
	def __init__(self, hidden_dims, scope):
		self.hidden_dims = hidden_dims
		self.scope = scope
        
	def __call__(self, states):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			hidden = states
			for hidden_dim in self.hidden_dims:
				hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
				                         kernel_initializer=tf.initializers.variance_scaling())
			hidden = tf.layers.dense(hidden, 1, activation=None,
			                       kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3))
			return hidden

class A2C(object):
    '''
    Implement off-policy A2C algorithm.
    Advantage is estimated as V(next)-V(current)+reward.
    '''
    def __init__(self, env, hidden_dims, replay_memory=None, gamma=1.0, tau=1e-2,
                 actor_lr=1e-3, critic_lr=1e-3, N=5, scope_pre = ""):
        self.env = env
        self.hidden_dims = hidden_dims
        self.state_dim = reduce(mul,env.observation_space.shape)
        self.n_action = env.action_space.n
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.N = N
        self.tau = tau
        self.scope_pre = scope_pre
        self.build()
    
    def build(self):
        self.build_actor_critic()
        self.build_placeholder()
        self.build_loss()
        self.build_step()
        self.build_summary()
    
    def build_actor_critic(self):
        self.actor = Actor_(self.hidden_dims, self.n_action, self.scope_pre+'actor')
        self.critic = VNetwork(self.hidden_dims, self.scope_pre+'critic')

    def build_placeholder(self):
        self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.actions = tf.placeholder(tf.int32, shape=[None,1])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.nexts = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.are_non_terminal = tf.placeholder(tf.float32, shape=[None])
        self.training = tf.placeholder(tf.bool)
    
    def build_loss(self):
        states = self.states
        nexts = self.nexts
        rewards = 0.01*self.rewards
        batch_size = tf.shape(states)[0]
        batch_indices = tf.range(batch_size,dtype=tf.int32)[:, None]
        target_value = tf.stop_gradient(rewards[:, None]+self.are_non_terminal[:, None] * \
                   np.power(self.gamma, self.N) * self.critic(nexts))
        value = self.critic(states)
        self.critic_loss = tf.losses.mean_squared_error(value, target_value)
        self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic")
        self.logits = self.actor(states)
        self.pi = tf.nn.softmax(self.logits, axis=-1)
        log_pi = tf.log(self.pi+1e-12)
        action_indices = tf.concat([batch_indices, self.actions], axis=1)
        log_action_prob = tf.gather_nd(log_pi, action_indices)[:, None]
        advantage = tf.stop_gradient(target_value-value)
        pg_loss = advantage*log_action_prob
        self.actor_loss = -tf.reduce_mean(pg_loss)

    def build_step(self):
        self.global_step = tf.Variable(0, trainable=False)
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
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'),
                global_step=self.global_step)
    
    def build_summary(self):
        tf.summary.scalar('critic_loss', self.critic_loss)
        self.merged_summary_op = tf.summary.merge_all()
    
    def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, epsilon=0.3, apply_her=False, n_goals=10):
        total_rewards = []
        for i_episode in tqdm(range(train_episodes), ncols=100):
            states, actions, returns, nexts, are_non_terminal, total_reward = self.collect_trajectory()
            append_summary(progress_fd, str(start_episode + i_episode) + ',{0:.2f}'.format(total_reward))
            total_rewards.append(total_reward)
            perm = np.random.permutation(len(states))
            for s in range(step):
            	sess.run([self.critic_step],
                     feed_dict={self.states: states,
                                self.actions: actions,
                                self.rewards: returns,
                                self.nexts: nexts,
                                self.are_non_terminal: are_non_terminal,
                                self.training: True})
            sess.run([self.actor_step],
                     feed_dict={self.states: states,
                                self.actions: actions,
                                self.rewards: returns,
                                self.nexts: nexts,
                                self.are_non_terminal: are_non_terminal,
                                self.training: True})
            # summary_writer.add_summary(summary, global_step=self.global_step.eval())
            if (i_episode + 1) % save_episodes == 0:
                saver.save(sess, model_path)
        return total_rewards
    
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
        
    def collect_trajectory(self):
        states, actions, rewards = self.generate_episode()
        states = np.array(states)
        returns, nexts, are_non_terminal = self.normalize_returns(states[1:],
                                                                  rewards)
        total_reward = sum(rewards)
        return states[:-1], actions, returns, nexts, are_non_terminal, total_reward
    
    def generate_episode(self, render=False):
        states = []
        actions = []
        rewards = []
        state = self.env.reset()
        while True:
            pi = self.pi.eval(feed_dict={
                self.states: np.array(state).reshape(1,-1),
                self.training: False}).squeeze(axis=0)      
            action = np.random.choice(np.arange(self.n_action), size=None,
                                      replace=True, p=pi)
            states.append(state)
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
        return states, actions, rewards