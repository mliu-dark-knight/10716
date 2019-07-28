from functools import reduce
from operator import mul
from typing import *
import gym
from tqdm import tqdm
from .A2C import Actor_
from .QRDDPG import QRCritic
from algorithms.common import *
from utils import append_summary

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
            mean = tf.layers.dense(hidden, self.action_dim, activation=tf.nn.tanh,
                                   kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3))
            log_std = tf.layers.dense(hidden, self.action_dim, activation=tf.nn.tanh,
                                   kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3))
            log_std = tf.clip_by_value(log_std, -20, 2)
            return mean, log_std


class SQRDDPG(object):
    def __init__(self, env, hidden_dims, replay_memory=None, gamma=1.0, init_std=0.2, tau=1e-2,
                 actor_lr=1e-3, critic_lr=1e-3, kappa=1, n_quantile=200, N=0, horrizon=256):
        self.env = env
        self.hidden_dims = hidden_dims
        self.state_dim = reduce(mul,self.env.observation_space.shape)
        self.kappa = kappa
        self.n_quantile = n_quantile
        self.horrizon = horrizon
        self.env_info = {'done': True, 'last_state': None, 'total_reward': 0}
        if isinstance(self.env.action_space, gym.spaces.box.Box):
            self.action_type = "continuous"
            self.n_action = self.env.action_space.shape[0]
            self.action_upper_limit = self.env.action_space.high
            self.action_lower_limit = self.env.action_space.low
        elif isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.action_type = "discrete"
            self.n_action = env.action_space.n
        else:
            raise NotImplementedError
        self.gamma = gamma
        self.init_std = init_std
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.N = N
        self.replay_memory = replay_memory
        self.alpha = tf.placeholder(tf.float32, shape=[1])
        self.build()
        
    
    def build(self):
        self.build_actor()
        self.build_critic()
        self.build_placeholder()
        self.build_loss()
        self.build_copy_op()
        self.build_step()

    def build_actor(self):
        self.actor = Actor(self.hidden_dims, self.n_action, 'actor')
        self.actor_target = Actor(self.hidden_dims, self.n_action, 'target_actor')

    def build_critic(self):
        self.critic_1 = QRCritic(self.hidden_dims, self.n_quantile,'critic_1')
        self.critic_target_1 = QRCritic(self.hidden_dims, self.n_quantile, 'target_critic_1')

        self.critic_2 = QRCritic(self.hidden_dims, self.n_quantile,'critic_2')
        self.critic_target_2 = QRCritic(self.hidden_dims, self.n_quantile, 'target_critic_2')
        

    def build_placeholder(self):
        self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.actions = tf.placeholder(tf.float32, shape=[None, self.n_action])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.nexts = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.are_non_terminal = tf.placeholder(tf.float32, shape=[None])
        self.training = tf.placeholder(tf.bool)

    def build_copy_op(self):
        self.init_critic_1, self.update_critic_1 = get_target_updates(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_1'),
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic_1'), self.tau)
        self.init_critic_2, self.update_critic_2 = get_target_updates(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_2'),
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic_2'), self.tau)
        #self.init_actor, self.update_actor = get_target_updates(
        #    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'),
        #    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor'), self.tau)

    def get_huber_quantile_regression_loss(self, Z1, Z2):
        errors = tf.expand_dims(Z1, axis=2) - tf.expand_dims(Z2, axis=1)
        huber_loss_case_one = tf.to_float(tf.abs(errors) <= self.kappa) * 0.5 * errors ** 2
        huber_loss_case_two = tf.to_float(tf.abs(errors) > self.kappa) * self.kappa * \
                              (tf.abs(errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two
        quantiles = tf.expand_dims(tf.expand_dims(tf.range(0.5 / self.n_quantile, 1., 1. / self.n_quantile), axis=0),
                                   axis=1)
        quantile_huber_loss = (tf.abs(quantiles - tf.stop_gradient(tf.to_float(errors < 0))) * huber_loss) / \
                              self.kappa
        return tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=2))

    def build_loss(self):
        #with tf.variable_scope('normalize_states'):
        #    bn = tf.layers.BatchNormalization(_reuse=tf.AUTO_REUSE)
            #states = bn.apply(self.states, training=self.training)
            #nexts = bn.apply(self.nexts, training=self.training)
        states = self.states
        nexts = self.nexts
        epsilon = tf.random.normal(shape=[tf.shape(states)[0], self.n_action])
        target_actor_output = self.actor(nexts)
        target_actions = target_actor_output[0] + epsilon*target_actor_output[1]
        entropy = 0.2*(self.get_action_loglikelihood(self.actor(states), self.actions))
        target_Z = tf.stop_gradient(tf.expand_dims(self.rewards, axis=1)-entropy+tf.expand_dims(self.are_non_terminal, axis=1) * \
                   np.power(self.gamma, self.N) * (self.critic_target_1(nexts, tf.tanh(target_actions))))
        Z = self.critic_1(states, tf.tanh(self.actions))
        self.critic_loss = self.get_huber_quantile_regression_loss(target_Z, Z)

        target_Z = tf.stop_gradient(tf.expand_dims(self.rewards, axis=1)-entropy+tf.expand_dims(self.are_non_terminal, axis=1) * \
                   np.power(self.gamma, self.N) * (self.critic_target_2(nexts, tf.tanh(target_actions))))
        Z = self.critic_2(states, tf.tanh(self.actions))
        self.critic_loss += self.get_huber_quantile_regression_loss(target_Z, Z)
        
        self.actor_output = self.actor(states)
        epsilon = tf.random.normal(shape=[tf.shape(states)[0], self.n_action])
        actions = self.actor_output[0] + epsilon*self.actor_output[1]
        
        qvalues_1 = tf.reduce_mean(self.critic_1(self.states, tf.tanh(actions)), axis=1)[:, None]
        qvalues_2 = tf.reduce_mean(self.critic_2(self.states, tf.tanh(actions)), axis=1)[:, None]
        self.qvalues = tf.minimum(qvalues_1, qvalues_2)
        #self.qvalues = tf.stop_gradient(tf.reduce_mean(self.critic(self.states, tf.tanh(actions)), axis=1)[:, None])
        #self.actor_loss = -tf.reduce_mean(self.qvalues)
        print_op = tf.print(self.alpha)
        #with tf.control_dependencies([print_op]):
        self.actor_loss = tf.reduce_mean(self.get_action_loglikelihood(self.actor_output, actions)-self.qvalues)

    def get_prob(self, value, Z):
        value = value[:, None]
        prob = 1e-10+tf.reduce_sum(tf.to_float(Z<=value), axis=1)[:, None]/self.n_quantile
        return prob

    def build_step(self):
        self.global_step = tf.Variable(0, trainable=False)
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.actor_loss), [self.actor_loss])]):
            self.actor_step = actor_optimizer.minimize(
                self.actor_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'))
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.critic_loss), [self.critic_loss])]):
            self.critic_step = critic_optimizer.minimize(
                self.critic_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_1')+tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_2'),
                global_step=self.global_step)

    def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, epsilon=0.3, apply_her=False, n_goals=10, train_steps=None):
        total_rewards = []
        total_steps = 0
        sess.run([self.init_critic_1, self.init_critic_2])
        epsilons = np.linspace(epsilon, 0.01, train_episodes)
        alphas = np.linspace(0.01, 1, train_episodes)
        for i_episode in tqdm(range(train_episodes), ncols=100):
            total_reward, values = self.collect_transitions(epsilons[i_episode], sess)
            total_steps += self.horrizon
            if not total_reward is None:
                append_summary(progress_fd, str(start_episode + i_episode) + ',{0:.2f}'.format(total_reward)+',{}'.format(total_steps)+',{0:.2f}'.format(values))
                total_rewards.append(total_reward)
            states, actions, rewards, nexts, are_non_terminal = self.replay_memory.sample_batch(step * batch_size)
            for t in range(step):
                feed_dict = {self.states: states[t * batch_size: (t + 1) * batch_size],
                                    self.actions: actions[t * batch_size: (t + 1) * batch_size],
                                    self.rewards: rewards[t * batch_size: (t + 1) * batch_size],
                                    self.nexts: nexts[t * batch_size: (t + 1) * batch_size],
                                    self.are_non_terminal: are_non_terminal[t * batch_size: (t + 1) * batch_size],
                                    self.training: True,
                                    self.alpha: np.array([alphas[i_episode]])}
                sess.run([self.critic_step, self.actor_step],
                         feed_dict=feed_dict)
                sess.run([self.update_critic_1, self.update_critic_2])
            actor_loss = sess.run(self.actor_loss, feed_dict=feed_dict)
            qvalues = sess.run([self.qvalues], feed_dict=feed_dict)
            print(actor_loss)
            # summary_writer.add_summary(summary, global_step=self.global_step.eval())
            if (i_episode + 1) % save_episodes == 0:
                saver.save(sess, model_path)
        return total_rewards

    def sample_action(self, states, sess):
        feed_dict = {self.states: states.reshape(1, -1), self.training: False}
        mean, log_std = sess.run(self.actor_output, feed_dict=feed_dict)
        action = np.random.normal(loc=mean[0], scale=np.exp(log_std[0]))
        return action

    def get_action_loglikelihood(self, actor_output, actions):
        mean, log_std = actor_output[0], actor_output[1]
        dst = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_std))
        action_loglikelihood = dst.log_prob(actions)
        action_loglikelihood -= tf.reduce_sum(tf.log(1e-6+1-tf.pow(tf.tanh(actions), 2)), axis=1)
        return action_loglikelihood[:, None]


    def collect_transitions(self, epsilon, sess):
        total_rewards = []
        states = np.zeros((self.horrizon+1, self.state_dim))
        if self.action_type == "continuous":
            actions = np.zeros((self.horrizon, self.n_action))
        else:
            actions = np.zeros((self.horrizon, 1))
        rewards = np.zeros(self.horrizon)
        are_non_terminal = np.zeros(self.horrizon)
        values = []
        for step in range(self.horrizon):
            if self.env_info['done']:
                state = self.env.reset()
                self.env_info['last_state'] = state
                self.env_info['done'] = False
                self.env_info['total_reward'] = 0
            state = self.env_info['last_state']
            action = self.sample_action(state, sess)
            states[step]=state
            state, reward, done, _ = self.env.step(np.tanh(action))
            rewards[step]=reward
            self.env_info['last_state'] = state
            self.env_info['total_reward'] += reward
            actions[step,:] = action
            if done:
                are_non_terminal[step] = 0
                self.env_info['done'] = True
                total_rewards.append(self.env_info['total_reward'])
                self.env_info['total_reward'] = 0
            else:
                are_non_terminal[step] = 1
        step += 1
        states[step]=state
        self.replay_memory.append(states[:-1],
                                  actions, rewards,
                                  states[1:], are_non_terminal)
        if len(total_rewards)>0:
            avg_rewards = np.mean(total_rewards)
            #avg_values = np.mean(values)
            avg_values = 0
        else:
            avg_rewards = None
            avg_values = None
        return avg_rewards, avg_values
    def collect_trajectory(self, epsilon):
        states, actions, rewards = self.generate_episode(epsilon=epsilon)
        states = np.array(states)
        rewards = np.array(rewards)
        nexts = states[1:]
        are_non_terminal = np.ones(len(nexts))
        are_non_terminal[-1] = 0
        total_reward = sum(rewards)
        self.replay_memory.append(states[:-1],
                                  actions, rewards,
                                  nexts, are_non_terminal)
        # hindsight experience
        return total_reward

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
            action = self.policy.eval(feed_dict={
                self.states: np.expand_dims(state, axis=0),
                self.training: False}).squeeze(axis=0)
            action = np.random.normal(loc=action, scale=epsilon)
            state, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()
            actions.append(action)
            rewards.append(reward)
            if done:
                break
        states.append(state)

        return states, actions, rewards
