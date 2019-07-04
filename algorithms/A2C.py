from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from algorithms.common import *
from utils import append_summary
import gym
import scipy

def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer

class Actor_(object):
    def __init__(self, hidden_dims, action_dim, scope):
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.scope = scope

    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.tanh,
                                         kernel_initializer=tf.initializers.orthogonal())
            return tf.layers.dense(hidden, self.action_dim, activation=None,
                                   kernel_initializer=tf.initializers.orthogonal())

class DirichletActor(object):
    def __init__(self, hidden_dims, action_dim, scope):
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.scope = scope

    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.tanh,
                                         kernel_initializer=tf.initializers.orthogonal())
            alpha = tf.layers.dense(hidden, self.action_dim, activation=tf.nn.softplus,
                                  kernel_initializer=tf.initializers.orthogonal())
            return alpha

class BetaActor(object):
    def __init__(self, hidden_dims, action_dim, scope):
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.scope = scope

    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.tanh,
                                         kernel_initializer=tf.initializers.orthogonal())
            alpha = tf.layers.dense(hidden, self.action_dim, activation=tf.nn.softplus,
                                  kernel_initializer=tf.initializers.orthogonal())
            beta = tf.layers.dense(hidden, self.action_dim, activation=tf.nn.softplus,
                                  kernel_initializer=tf.initializers.orthogonal())
            return alpha+1, beta+1

class GaussianActor(object):
    def __init__(self, hidden_dims, action_dim, scope):
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.scope = scope
        with tf.variable_scope(scope):
            self.log_std = tf.get_variable(name=self.scope+"_log_std", shape=[1, self.action_dim], initializer=tf.zeros_initializer())

    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         kernel_initializer=tf.initializers.orthogonal())
            mean = tf.layers.dense(hidden, self.action_dim, activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         kernel_initializer=tf.initializers.orthogonal())
            #log_std = tf.layers.dense(hidden, self.action_dim, activation=None,
            #                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            #                             bias_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            #                             kernel_initializer=tf.initializers.orthogonal())
            #log_std = tf.clip_by_value(log_std, -20, 2)
            batch_size = tf.shape(states)[0]
            b = tf.ones((batch_size, 1))
            log_std = tf.tile(self.log_std, (batch_size, 1))
            log_std = self.log_std*b
            return mean, log_std

class VNetwork(object):
    def __init__(self, hidden_dims, scope):
        self.hidden_dims = hidden_dims
        self.scope = scope
        
    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.orthogonal(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
            hidden = tf.layers.dense(hidden, 1, activation=None,
                                   kernel_initializer=tf.initializers.orthogonal(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
            return hidden

class A2C(object):
    '''
    Implement A2C algorithm.
    Advantage is estimated as V(next)-V(current)+reward.
    '''
    def __init__(self, env, hidden_dims, gamma=0.99, tau=1e-2,
                 actor_lr=1e-3, critic_lr=1e-3, N=5, scope_pre = "", is_env_pool=False):
        self.env = env
        self.is_env_pool = is_env_pool
        if self.is_env_pool:
            self.env_pool = self.env
            self.env = self.env_pool.sample_env()
        self.hidden_dims = hidden_dims
        self.state_dim = reduce(mul,self.env.observation_space.shape)
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
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.N = N
        self.tau = tau
        self.scope_pre = scope_pre
        self.build()
    
    def build(self):
        self.build_actor()
        self.build_critic()
        self.build_placeholder()
        self.build_loss()
        self.build_step()
        self.build_summary()
        self.build_copy_op()

    def build_copy_op(self):
        self.init_critic, self.update_critic = get_target_updates(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'),
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic'), self.tau)
    
    def build_actor(self):
        if self.action_type == "discrete":
            self.actor = Actor_(self.hidden_dims, self.n_action, self.scope_pre+'actor')
        else:
            self.actor = BetaActor(self.hidden_dims, self.n_action, self.scope_pre+'actor')

    def build_critic(self):
        self.critic = VNetwork(self.hidden_dims, self.scope_pre+'critic')
        self.critic_target = VNetwork(self.hidden_dims, self.scope_pre+'target_critic')

    def build_placeholder(self):
        self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        if self.action_type == "continuous":
            self.actions = tf.placeholder(tf.float32, shape=[None, self.n_action])
        else:
            self.actions = tf.placeholder(tf.int32, shape=[None, 1])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.nexts = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.are_non_terminal = tf.placeholder(tf.float32, shape=[None])
        self.training = tf.placeholder(tf.bool)
    
    def get_action_loglikelihood(self, actor_output, actions):
        batch_size = tf.shape(actor_output[0])[0]
        if isinstance(self.actor, BetaActor):
            e = tf.ones((batch_size, self.n_action))*1e-12
            alpha, beta = actor_output[0], actor_output[1]
            action_loglikelihood = (alpha-1.)*tf.log(actions+e)+(beta-1.)*tf.log(1-actions+e)
            action_loglikelihood += -tf.math.lgamma(alpha)-tf.math.lgamma(beta)+tf.math.lgamma(alpha+beta)
            action_loglikelihood = tf.reduce_sum(action_loglikelihood, axis=1)[:, None]
        elif isinstance(self.actor, Actor_):
            e = (tf.ones(batch_size)*1e-12)[:, None]
            batch_indices = tf.range(batch_size,dtype=tf.int32)[:, None]
            action_indices = tf.concat([batch_indices, actions], axis=1)
            logits = actor_output
            logpi = tf.log(tf.nn.softmax(logits, axis=-1)+e)
            action_loglikelihood = tf.gather_nd(logpi, action_indices)
        elif isinstance(self.actor, GaussianActor):
            mean, log_std = actor_output[0], actor_output[1]
            std = tf.exp(log_std)
            #actions_ = tf.tanh(actions)
            action_loglikelihood = -0.5*tf.log(2*np.pi)-log_std
            action_loglikelihood += -0.5*tf.pow((actions-mean)/std, 2)
            #action_loglikelihood -= tf.log(1-tf.pow(tf.tanh(actions_), 2))
            #action_loglikelihood -= tf.log(1-tf.pow(actions_, 2)+1e-6)
            action_loglikelihood = tf.reduce_sum(action_loglikelihood, axis=1)[:, None]
        else:
            raise NotImplementedError
        return action_loglikelihood
    
    def get_policy_entropy(self, actor_output):
        batch_size = tf.shape(actor_output[0])[0]
        if isinstance(self.actor, BetaActor):
            alpha, beta = actor_output[0], actor_output[1]
            entropy = tf.math.lbeta(tf.concat([alpha, beta], axis=1))[:, None]
            entropy -= (alpha-1.)*tf.math.digamma(alpha)
            entropy -= (beta-1.)*tf.math.digamma(beta)
            entropy += (alpha+beta-2)*tf.math.digamma(alpha+beta)
            return entropy
        elif isinstance(self.actor, Actor_):
            logits = actor_output
            pi = tf.nn.softmax(logits, axis=-1)
            logpi = tf.log(tf.nn.softmax(logits, axis=-1)+e)
            entropy = -tf.reduce_sum(pi*logpi, axis=1)[:, None]
            return entropy
        elif isinstance(self.actor, GaussianActor):
            mean, log_std = actor_output[0], actor_output[1]
            entropy = tf.log(tf.sqrt(2*np.pi*np.e))+log_std
            return entropy
        else:
            raise NotImplementedError

    def convert_action_for_beta_actor(self, actions):
        actions -= self.action_lower_limit
        actions /= self.action_upper_limit - self.action_lower_limit
        return actions

    def sample_action(self, states):
        feed_dict = {self.states: states.reshape(1, -1), self.training: False}
        
        if isinstance(self.actor, BetaActor):
            alpha = self.actor_output[0].eval(feed_dict=feed_dict).squeeze(axis=0)
            beta = self.actor_output[1].eval(feed_dict=feed_dict).squeeze(axis=0)
            action = np.random.beta(alpha, beta)
            e = 1e-12
            action_loglikelihood = (alpha-1.)*np.log(action+e)+(beta-1.)*np.log(1-action)
            action_loglikelihood += -scipy.special.loggamma(alpha)-scipy.special.loggamma(beta)+scipy.special.loggamma(alpha+beta)
            action_loglikelihood = np.sum(action_loglikelihood)
            action *= self.action_upper_limit - self.action_lower_limit
            action += self.action_lower_limit
        elif isinstance(self.actor, Actor_):
            logits = self.actor_output.eval(feed_dict=feed_dict).squeeze(axis=0)
            pi = np.exp(logits - np.max(logits))
            pi /= np.sum(pi)
            action = np.random.choice(np.arange(self.n_action), size=None,
                                      replace=True, p=pi)
            action_loglikelihood = np.log(pi[:, action]+1e-9)
        elif isinstance(self.actor, GaussianActor):
            mean = self.actor_output[0].eval(feed_dict=feed_dict).squeeze(axis=0)
            log_std = self.actor_output[1].eval(feed_dict=feed_dict).squeeze(axis=0)
            std = np.exp(log_std)
            action = np.random.normal(loc=mean, scale=std)
            action_ = np.tanh(action)
            action_loglikelihood = -0.5*np.log(2*np.pi)-log_std
            action_loglikelihood += -0.5*np.power((action-mean)/std, 2)
            action_loglikelihood = np.sum(action_loglikelihood)
            #action_loglikelihood -= np.sum(np.log(1-np.power(action_, 2)+1e-6))
        return action, action_loglikelihood

    def build_loss(self):
        states = self.states
        nexts = self.nexts
        rewards = self.rewards
        batch_size = tf.shape(states)[0]
        batch_indices = tf.range(batch_size,dtype=tf.int32)[:, None]
        target_value = tf.stop_gradient(rewards[:, None]+self.are_non_terminal[:, None] * \
                   np.power(self.gamma, self.N) * self.critic_target(nexts))
        value = self.critic(states)
        #self.critic_loss = tf.losses.mean_squared_error(value, target_value)
        self.critic_loss = tf.losses.huber_loss(target_value, value)
        self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic")
        self.actor_output = self.actor(states)
        action_loglikelihood = self.get_action_loglikelihood(self.actor_output, self.actions)
        advantage = tf.stop_gradient(target_value-value)
        pg_loss = advantage*action_loglikelihood
        self.actor_loss = -tf.reduce_mean(pg_loss)

    def build_step(self):
        def clip_grad_by_global_norm(grad_var, max_norm):
            grad_var = list(zip(*grad_var))
            grad, var = grad_var[0], grad_var[1]
            clipped_grad,_ = tf.clip_by_global_norm(grad, max_norm)
            return list(zip(clipped_grad, var))

        self.global_step = tf.Variable(0, trainable=False)
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.actor_loss), [self.actor_loss])]):
            #gvs = actor_optimizer.compute_gradients(self.actor_loss,
            #   var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'))
            #clipped_grad_var = clip_grad_by_global_norm(gvs, 5)
            #self.actor_step = actor_optimizer.apply_gradients(clipped_grad_var)
            self.actor_step = actor_optimizer.minimize(
                self.actor_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'))
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.critic_loss), [self.critic_loss])]):
            self.critic_step = critic_optimizer.minimize(
                self.critic_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'),
                global_step=self.global_step)
            #gvs = critic_optimizer.compute_gradients(self.critic_loss,
            #   var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))
            #clipped_grad_var = clip_grad_by_global_norm(gvs, 1)
            #self.critic_step = actor_optimizer.apply_gradients(clipped_grad_var)
    
    def build_summary(self):
        tf.summary.scalar('critic_loss', self.critic_loss)
        self.merged_summary_op = tf.summary.merge_all()
    
    def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, epsilon=0.3, apply_her=False, n_goals=10):
        total_rewards = []
        sess.run([self.init_critic])
        for i_episode in tqdm(range(train_episodes), ncols=100):
            states, actions, returns, nexts, are_non_terminal, total_reward = self.collect_trajectory()
            feed_dict = {self.states: states, self.actions: actions, self.rewards: returns,
                         self.nexts: nexts, self.are_non_terminal: are_non_terminal, self.training: True}
            total_rewards.append(total_reward)
            perm = np.random.permutation(len(states))
            for s in range(step):
                sess.run([self.critic_step],
                     feed_dict=feed_dict)
            sess.run([self.update_critic])
            sess.run([self.actor_step],
                     feed_dict=feed_dict)
            
            # summary_writer.add_summary(summary, global_step=self.global_step.eval())
            critic_loss = self.critic_loss.eval(feed_dict=feed_dict).mean()
            actor_loss = self.actor_loss.eval(feed_dict=feed_dict).mean()
            append_summary(progress_fd, str(start_episode + i_episode) + ",{0:.2f}".format(total_reward)\
                +",{0:.4f}".format(actor_loss)+",{0:.4f}".format(critic_loss))
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
        #returns, nexts, are_non_terminal = self.normalize_returns(states[1:],
        #                                                         rewards)
        nexts = states[1:].copy()
        are_non_terminal = np.ones(len(nexts))
        are_non_terminal[-1] = 0
        total_reward = sum(rewards)
        #return states[:-1], actions, returns, nexts, are_non_terminal, total_reward
        return states[:-1], actions, rewards, nexts, are_non_terminal, total_reward
    
    def generate_episode(self, render=False):
        states = []
        actions = []
        rewards = []
        if self.is_env_pool:
            self.env = self.env_pool.sample_env()
        state = self.env.reset()
        while True:
            action = self.sample_action(state)
            states.append(state)
            state, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()
            if self.action_type == "discrete":
                actions.append([action])
            else:
                actions.append(action)
            rewards.append(reward)
            if done:
                break
        states.append(state)

        actions = np.array(actions)
        if isinstance(self.actor, BetaActor):
            actions = self.convert_action_for_beta_actor(actions)
        assert len(states)  == len(actions)+1 and \
               len(actions) == len(rewards)
        return states, actions, rewards