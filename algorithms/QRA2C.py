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
                                                                                             mode='fan_avg'))
            hidden = tf.layers.dense(hidden, self.n_quantile, activation=None,
                                   kernel_initializer=tf.initializers.variance_scaling(distribution='uniform',
                                                                                             mode='fan_avg'))
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
    
    def build_actor_critic(self):
        self.actor = Actor_(self.hidden_dims, self.n_action, self.scope_pre+'actor')
        self.critic = QRVNetwork(self.hidden_dims, self.n_quantile, self.scope_pre+'critic')
        self.critic_target = QRVNetwork(self.hidden_dims, self.n_quantile, self.scope_pre+'target_critic')

    def build_loss(self):
        with tf.variable_scope(self.scope_pre+'normalize_states'):
            bn = tf.layers.BatchNormalization(_reuse=tf.AUTO_REUSE)
            states = bn.apply(self.states[:, :-1], training=self.training)
            nexts = bn.apply(self.nexts[:, :-1], training=self.training)
        prob = self.states[:, -1]
        batch_size = tf.shape(states)[0]
        batch_indices = tf.range(batch_size,dtype=tf.int32)[:, None]
        
        target_Z = tf.stop_gradient(self.rewards[:,None]+self.are_non_terminal[:, None] * \
                   np.power(self.gamma, self.N) * self.critic_target(nexts))
        Z = self.critic(states)
        bellman_errors = tf.expand_dims(target_Z, axis=2) - tf.expand_dims(Z, axis=1)
        huber_loss_case_one = tf.to_float(tf.abs(bellman_errors) <= self.kappa) * 0.5 * bellman_errors ** 2
        huber_loss_case_two = tf.to_float(tf.abs(bellman_errors) > self.kappa) * self.kappa * \
                              (tf.abs(bellman_errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two
        quantiles = tf.expand_dims(tf.expand_dims(tf.range(0.5 / self.n_quantile, 1., 1. / self.n_quantile), axis=0),
                                   axis=1)
        quantile_huber_loss = (tf.abs(quantiles - tf.stop_gradient(tf.to_float(bellman_errors < 0))) * huber_loss) / \
                              self.kappa
        self.critic_loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=2), axis=1), axis=0)
        self.logits = self.actor(states)
        self.pi = tf.nn.softmax(self.logits, axis=-1)
        log_pi = tf.log(self.pi+1e-9)
        action_indices = tf.concat([batch_indices, self.actions], axis=1)
        log_action_prob = tf.gather_nd(log_pi, action_indices)/prob
        EA = tf.stop_gradient(self.rewards
                              +tf.reduce_mean(self.critic(nexts), axis=1)
                              -tf.reduce_mean(self.critic(states), axis=1))
        self.actor_loss = -tf.reduce_mean(EA*log_action_prob)