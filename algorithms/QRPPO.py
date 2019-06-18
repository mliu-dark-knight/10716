from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from algorithms.common import *
from algorithms.PPO import PPO
from algorithms.A2C import Actor_, BetaActor
from algorithms.QRA2C import QRVNetwork, QRVNetworkNoCrossing
from utils import append_summary
import gym

class QRPPO(PPO):
    '''
    Implement off-policy A2C algorithm.
    Advantage is estimated as V(next)-V(current)+reward.
    '''
    def __init__(self, *args,kappa=1.0, n_quantile=64, **kwargs):
        self.kappa = kappa
        self.n_quantile = n_quantile
        super(QRPPO, self).__init__(*args, **kwargs)

    def build_actor(self):
        if self.action_type == "discrete":
            self.actor = Actor_(self.hidden_dims, self.n_action, self.scope_pre+'actor')
            self.actor_target = Actor_(self.hidden_dims, self.n_action, self.scope_pre+'target_actor')
        else:
            self.actor = BetaActor(self.hidden_dims, self.n_action, self.scope_pre+'actor')
            self.actor_target = BetaActor(self.hidden_dims, self.n_action, self.scope_pre+'target_actor')

    def build_critic(self):
        self.critic = QRVNetworkNoCrossing(self.hidden_dims, self.n_quantile, self.scope_pre+'critic')
        #self.critic_target = QRVNetworkNoCrossing(self.hidden_dims, self.n_quantile, self.scope_pre+'target_critic')

    def build_copy_op(self):
        self.init_actor, self.update_actor = get_target_updates(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'),
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor'), 1)
        #self.init_critic, self.update_critic = get_target_updates(
        #    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'),
        #    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic'), self.tau)
    
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

    '''def build_critic_loss(self):
        self.target_Z = tf.stop_gradient(self.rewards[:,None]+self.are_non_terminal[:, None] * \
                   np.power(self.gamma, self.N) * self.critic_target(self.nexts))
        self.Z = self.critic(states)
        regression_loss = self.get_huber_quantile_regression_loss(self.target_Z, self.Z)
        self.critic_loss = tf.reduce_mean(self.get_huber_quantile_regression_loss(self.target_Z, self.Z), axis=0)
        self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic")'''

    def build_critic_loss(self):
        self.compute_return()
        self.Z = self.critic(self.states)
        errors = self.returns - self.Z
        huber_loss_case_one = tf.to_float(tf.abs(errors) <= self.kappa) * 0.5 * errors ** 2
        huber_loss_case_two = tf.to_float(tf.abs(errors) > self.kappa) * self.kappa * \
                              (tf.abs(errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two
        quantiles = tf.expand_dims(tf.range(0., 1., 1. / self.n_quantile), axis=0)
        quantile_huber_loss = (tf.abs(quantiles - tf.stop_gradient(tf.to_float(errors < 0))) * huber_loss) / \
                              self.kappa
        self.critic_loss = tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=1), axis=0)
        #self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic")
        self.target_Z =  tf.stop_gradient(self.rewards[:,None]+self.are_non_terminal[:, None] * \
                   np.power(self.gamma, self.N) * self.critic(self.nexts))
        self.target_value = self.get_mean(self.target_Z)
        self.value = self.get_mean(self.Z)
        self.critic_loss += tf.losses.huber_loss(self.returns, self.value[:, None])

    def build_loss(self):
        batch_size = tf.shape(self.states)[0]
        batch_indices = tf.range(batch_size,dtype=tf.int32)[:, None]
        self.build_critic_loss()        
        
        self.actor_output = self.actor(self.states)
        action_loglikelihood = self.get_action_loglikelihood(self.actor_output, self.actions)
        old_action_loglikelihood = self.get_action_loglikelihood(self.actor_target(self.states), self.actions)
        ratio = tf.exp(action_loglikelihood-old_action_loglikelihood)
        advantage = tf.stop_gradient(self.compute_gae(self.value[:, None], self.target_value[:, None]))
        pg_loss = tf.minimum(ratio*advantage, tf.clip_by_value(ratio, 0.8, 1.2)*advantage)
        
        entropy = self.get_policy_entropy(self.actor_output)
        self.actor_loss = -tf.reduce_mean(pg_loss+1e-2*entropy)