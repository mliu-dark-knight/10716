from functools import reduce
from operator import mul
from typing import *

from tqdm import tqdm

from algorithms.common import *
from utils import append_summary


class DDPG(object):

	def __init__(self, env, hidden_dims, replay_memory=None, gamma=1.0, init_std=0.2, tau=1e-2,
	             actor_lr=1e-3, critic_lr=1e-3, N=5):
		self.env = env
		self.goal_dim = reduce(mul, env.observation_space.spaces['desired_goal'].shape)
		self.state_dim = reduce(mul, env.observation_space.spaces['observation'].shape) + self.goal_dim
		self.action_dim = reduce(mul, env.action_space.shape)

		self.gamma = gamma
		self.init_std = init_std
		self.tau = tau
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		self.N = N

		self.actor = Actor(hidden_dims, self.action_dim, 'actor')
		self.actor_target = Actor(hidden_dims, self.action_dim, 'target_actor')
		self.critic = Critic(hidden_dims, 'critic')
		self.critic_target = Critic(hidden_dims, 'target_critic')
		self.replay_memory = replay_memory
		self.build()

	def build(self):
		self.build_placeholder()
		self.build_loss()
		self.build_copy_op()
		self.build_step()
		self.build_summary()

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
		           tf.squeeze(self.critic_target(nexts, self.actor_target(nexts)), axis=1)
		Q = tf.squeeze(self.critic(states, self.actions), axis=1)
		self.critic_loss = tf.losses.mean_squared_error(Q_target, Q)
		self.policy = self.actor(states)
		self.actor_loss = -tf.reduce_mean(self.critic(states, self.policy))

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

	def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, train_episodes=1000,
	          save_episodes=100,
	          epsilon=0.3):
		total_rewards = []
		sess.run([self.init_actor, self.init_critic])
		for i_episode in tqdm(range(train_episodes), ncols=100):
			total_reward = self.collect_trajectory(epsilon)
			append_summary(progress_fd, str(i_episode) + ',{0:.4f}'.format(total_reward))
			total_rewards.append(total_reward)
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
			# summary_writer.add_summary(summary, global_step=self.global_step.eval())
			if (i_episode + 1) % save_episodes == 0:
				saver.save(sess, model_path)
		return total_rewards

	def apply_her(self, states: np.array, achieved_goals: np.array, actions: List):
		desired_goals = np.zeros_like(achieved_goals)
		for T in range(len(achieved_goals)):
			goal = achieved_goals[T]
			desired_goals[:T + 1, ] = goal
			rewards = self.env.compute_reward(achieved_goals[:T + 1], desired_goals[:T + 1], None)
			returns, nexts, are_non_terminal = self.normalize_returns(states[1:T + 2], list(rewards))
			self.replay_memory.append(list(concat_state_goal(states[:T + 1], desired_goals[:T + 1])),
			                          actions[:T + 1], returns,
			                          list(concat_state_goal(nexts, desired_goals[:T + 1])),
			                          are_non_terminal)

	def collect_trajectory(self, epsilon):
		states, achieved_goals, desired_goals, actions, rewards = self.generate_episode(epsilon=epsilon)
		states, achieved_goals, desired_goals = np.array(states), np.array(achieved_goals), np.array(desired_goals)
		returns, nexts, are_non_terminal = self.normalize_returns(states[1:], rewards)
		total_reward = sum(rewards)
		self.replay_memory.append(list(concat_state_goal(states[:-1], desired_goals)),
		                          actions, returns,
		                          list(concat_state_goal(nexts, desired_goals)),
		                          are_non_terminal)

		# hindsight experience
		self.apply_her(states, achieved_goals, actions)
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

		return list(returns), nexts, list(are_non_terminal)

	def generate_episode(self, epsilon=0.0, render=False):
		'''
		:param epsilon: exploration noise
		:return:
		WARNING: make sure whatever you return comes from the same episode
		'''
		states = []
		achieved_goals = []
		desired_goals = []
		actions = []
		rewards = []

		triple = self.env.reset()
		state = triple[OBSERVATION_KEY]

		while True:
			states.append(state)
			desired_goals.append(triple[DESIRED_GOAL_KEY])
			action = self.policy.eval(feed_dict={
				self.states: np.expand_dims(np.concatenate((state, triple[DESIRED_GOAL_KEY]), axis=0), axis=0),
				self.training: False}).squeeze(axis=0)
			action = np.random.normal(loc=action, scale=epsilon)
			triple, reward, done, _ = self.env.step(action)
			if render:
				self.env.render()
			next = triple[OBSERVATION_KEY]
			achieved_goals.append(triple[ACHIEVED_GOAL_KEY])
			actions.append(action)
			rewards.append(reward)
			state = next
			if done:
				break
		states.append(state)
		assert len(states) == len(achieved_goals) + 1 and \
		       len(achieved_goals) == len(desired_goals) and \
		       len(desired_goals) == len(actions) and \
		       len(actions) == len(rewards)

		return states, achieved_goals, desired_goals, actions, rewards
