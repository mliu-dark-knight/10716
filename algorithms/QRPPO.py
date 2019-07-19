from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from algorithms.common import *
from algorithms.PPO import PPO
from algorithms.A2C import Actor_, BetaActor, GaussianActor
from algorithms.QRA2C import QRVNetwork, QRVNetworkNoCrossing
from utils import append_summary
import gym

class QRPPO(PPO):
    '''
    Implement off-policy A2C algorithm.
    Advantage is estimated as V(next)-V(current)+reward.
    '''
    def __init__(self, *args,kappa=1.0, n_quantile=64, entropy_scale=0.2, **kwargs):
        self.kappa = kappa
        self.n_quantile = n_quantile
        self.entropy_scale = entropy_scale
        super(QRPPO, self).__init__(*args, **kwargs)

    def build_actor(self):
        if self.action_type == "discrete":
            self.actor = Actor_(self.hidden_dims, self.n_action, self.scope_pre+'actor')
        else:
            self.actor = GaussianActor(self.hidden_dims, self.n_action, self.scope_pre+'actor')

    def build_critic(self):
        #self.critic_0 = QRVNetworkNoCrossing(self.hidden_dims, self.n_quantile, self.scope_pre+'critic_0')
        #self.critic_1 = QRVNetworkNoCrossing(self.hidden_dims, self.n_quantile, self.scope_pre+'critic_1')
        self.critic = QRVNetworkNoCrossing(self.hidden_dims, self.n_quantile, self.scope_pre+'critic')
    
    def get_mean(self, Z):
        part1 = Z[:, :-2:2]
        part2 = Z[:,1:-1:2]
        part3 = Z[:, 2::2]
        Z = (part1+4.*part2+part3)/3.
        return tf.reduce_mean(Z, axis=1)
    
    def build_critic_loss(self):
        self.Z = self.critic(self.states)
        #self.values = self.get_mean(self.Z)[:, None]
        self.values = self.Z[:, int(self.n_quantile/2)][:, None]
        errors = self.returns - self.Z
        huber_loss_case_one = tf.to_float(tf.abs(errors) <= self.kappa) * 0.5 * errors ** 2
        huber_loss_case_two = tf.to_float(tf.abs(errors) > self.kappa) * self.kappa * \
                              (tf.abs(errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two
        quantiles = tf.expand_dims(tf.linspace(1. / self.n_quantile, 1-1. / self.n_quantile, self.n_quantile), axis=0)
        quantile_huber_loss = (tf.abs(quantiles - tf.stop_gradient(tf.to_float(errors < 0))) * huber_loss) / \
                              self.kappa
        self.critic_loss = tf.reduce_mean(tf.reduce_mean(quantile_huber_loss, axis=1), axis=0)
        self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic")
        return_entropy = self.get_mean(tf.log(1e-10+self.critic.dQ(self.states)[1]))[:, None]
        self.critic_loss -= self.entropy_scale*tf.reduce_mean(return_entropy)

    '''def build_critic_loss(self):
        Z_0 = self.critic_0(self.states)
        values_0 = Z_0[:, int(self.n_quantile/2)]
        errors = self.returns - Z_0
        huber_loss_case_one = tf.to_float(tf.abs(errors) <= self.kappa) * 0.5 * errors ** 2
        huber_loss_case_two = tf.to_float(tf.abs(errors) > self.kappa) * self.kappa * \
                              (tf.abs(errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two
        quantiles = tf.expand_dims(tf.linspace(1. / self.n_quantile, 1-1. / self.n_quantile, self.n_quantile), axis=0)
        quantile_huber_loss = (tf.abs(quantiles - tf.stop_gradient(tf.to_float(errors < 0))) * huber_loss) / \
                              self.kappa
        self.critic_loss = tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=1), axis=0)
        self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic_0")
        return_entropy = self.get_mean(tf.log(1e-10+self.critic_0.dQ(self.states)[1]))[:, None]
        self.critic_loss -= tf.reduce_mean(return_entropy)

        Z_1 = self.critic_1(self.states)
        values_1 = Z_1[:, int(self.n_quantile/2)]
        errors = self.returns - Z_1
        huber_loss_case_one = tf.to_float(tf.abs(errors) <= self.kappa) * 0.5 * errors ** 2
        huber_loss_case_two = tf.to_float(tf.abs(errors) > self.kappa) * self.kappa * \
                              (tf.abs(errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two
        quantiles = tf.expand_dims(tf.linspace(1. / self.n_quantile, 1-1. / self.n_quantile, self.n_quantile), axis=0)
        quantile_huber_loss = (tf.abs(quantiles - tf.stop_gradient(tf.to_float(errors < 0))) * huber_loss) / \
                              self.kappa
        self.critic_loss += tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=1), axis=0)
        self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic_1")
        return_entropy = self.get_mean(tf.log(1e-10+self.critic_1.dQ(self.states)[1]))[:, None]
        self.critic_loss -= tf.reduce_mean(return_entropy)

        self.values = tf.minimum(values_0, values_1)'''      
        