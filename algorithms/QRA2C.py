from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from algorithms.common import *
from algorithms.A2C import A2C, Actor_
from utils import append_summary

class QRVNetwork(object):
    def __init__(self, hidden_dims, n_quantile, scope):
        self.hidden_dims = hidden_dims
        self.n_quantile = n_quantile
        self.scope = scope
    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.tanh,
                                         kernel_initializer=tf.initializers.orthogonal())
            
            hidden = tf.layers.dense(hidden, self.n_quantile, activation=None,
                                     kernel_initializer=tf.initializers.orthogonal())
            return hidden

class QRVNetworkNoCrossing(object):
    def __init__(self, hidden_dims, n_quantile, scope):
        self.hidden_dims = hidden_dims
        self.n_quantile = n_quantile
        self.scope = scope
    def __call__(self, states):
        base, quantiles = self.dQ(states)
        out = tf.concat([base, quantiles], axis=1)
        out = tf.math.cumsum(out, axis=1)
        return out
        
    def dQ(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         kernel_initializer=tf.initializers.orthogonal())
            base = tf.layers.dense(hidden, 1, activation=None,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         kernel_initializer=tf.initializers.orthogonal())
            quantiles = tf.layers.dense(hidden, self.n_quantile-1, activation=tf.nn.relu,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         kernel_initializer=tf.initializers.orthogonal())
        return base, quantiles

class QRA2C(A2C):
    '''
    Implement off-policy A2C algorithm.
    Critic is trained with quantile regression algorithm.
    Advantage is estimated as V(next)-V(current)+reward.
    '''
    def __init__(self, *args, kappa=1.0, n_quantile=64, **kwargs):
        self.kappa = kappa
        self.n_quantile = n_quantile
        super(QRA2C, self).__init__(*args, **kwargs)

    def build_critic(self):
        self.critic = QRVNetwork(self.hidden_dims, self.n_quantile, self.scope_pre+'critic')

    def get_mean(self, Z):
        part1 = Z[:, :-2:2]
        part2 = Z[:,1:-1:2]
        part3 = Z[:, 2::2]
        Z = (part1+4.*part2+part3)/3.
        return tf.reduce_mean(Z, axis=1)

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
        return tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=2), axis=1)

    def build_loss(self):
        states = self.states
        nexts = self.nexts
        rewards = self.rewards
        batch_size = tf.shape(states)[0]
        batch_indices = tf.range(batch_size,dtype=tf.int32)[:, None]
        target_Z = tf.stop_gradient(rewards[:,None]+self.are_non_terminal[:, None] * \
                   np.power(self.gamma, self.N) * self.critic(nexts))
        Z = self.critic(states)
        regression_loss = self.get_huber_quantile_regression_loss(target_Z, Z)
        self.critic_loss = tf.reduce_mean(self.get_huber_quantile_regression_loss(target_Z, Z), axis=0)
        self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic")
        
        self.actor_output = self.actor(states)
        action_loglikelihood = self.get_action_loglikelihood(self.actor_output, self.actions)
        advantage = tf.stop_gradient(self.get_mean(target_Z)-self.get_mean(Z))
        pg_loss = advantage*action_loglikelihood
        self.actor_loss = -tf.reduce_mean(pg_loss)

    def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, epsilon=0.3, apply_her=False, n_goals=10):
        total_rewards = []
        sess.run([self.init_actor, self.init_critic])
        for i_episode in tqdm(range(train_episodes), ncols=100):
            states, actions, returns, nexts, are_non_terminal, total_reward = self.collect_trajectory()
            feed_dict = {self.states: states, self.actions: actions, self.rewards: returns,
                         self.nexts: nexts, self.are_non_terminal: are_non_terminal, self.training: True}
            total_rewards.append(total_reward)
            perm = np.random.permutation(len(states))
            for s in range(step):
                sess.run([self.critic_step], feed_dict=feed_dict)
                sess.run([self.actor_step], feed_dict=feed_dict)
            sess.run([self.update_critic])
            sess.run([self.update_actor])
            # summary_writer.add_summary(summary, global_step=self.global_step.eval())
            critic_loss = self.critic_loss.eval(feed_dict=feed_dict).mean()
            actor_loss = self.actor_loss.eval(feed_dict=feed_dict).mean()
            append_summary(progress_fd, str(start_episode + i_episode) + ",{0:.2f}".format(total_reward)\
                +",{0:.4f}".format(actor_loss)+",{0:.4f}".format(critic_loss))
            if (i_episode + 1) % save_episodes == 0:
                saver.save(sess, model_path)
        return total_rewards