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
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.variance_scaling(distribution='uniform',
                                                                                             mode='fan_avg'),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
            hidden = tf.layers.dense(hidden, self.n_quantile, activation=None,
                                   kernel_initializer=tf.initializers.variance_scaling(distribution='uniform',mode='fan_avg'),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
            return hidden


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
        Z_old = tf.stop_gradient(self.critic(states))
        regression_loss = self.get_huber_quantile_regression_loss(target_Z, Z)
        self.critic_loss = tf.reduce_mean(self.get_huber_quantile_regression_loss(target_Z, Z), axis=0)
        self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic")
        
        self.actor_output = self.actor(states)
        action_loglikelihood = self.get_action_loglikelihood(self.actor_output, self.actions)
        advantage = tf.stop_gradient(self.get_mean(target_Z)-self.get_mean(Z))
        pg_loss = advantage*action_loglikelihood
        self.actor_loss = -tf.reduce_mean(pg_loss)