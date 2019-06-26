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

class SQRPPO(PPO):
    def __init__(self, *args,kappa=1.0, quantile=0.9, **kwargs):
        self.kappa = kappa
        self.n_quantile = 1
        self.quantile = quantile
        super(SQRPPO, self).__init__(*args, **kwargs)

    def build_critic(self):
        self.critic = QRVNetworkNoCrossing(self.hidden_dims, self.n_quantile, self.scope_pre+'critic')

    def build_critic_loss(self):
        self.Z = self.critic(self.states)
        self.values = self.Z

        errors = self.returns - self.Z
        huber_loss_case_one = tf.to_float(tf.abs(errors) <= self.kappa) * 0.5 * errors ** 2
        huber_loss_case_two = tf.to_float(tf.abs(errors) > self.kappa) * self.kappa * \
                              (tf.abs(errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two
        quantile_huber_loss = (tf.abs(self.quantile - tf.stop_gradient(tf.to_float(errors < 0))) * huber_loss) / \
                              self.kappa
        self.critic_loss = tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=1), axis=0)
        self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic")

    def build_actor_loss(self):
        self.actor_output = self.actor(self.states)
        action_loglikelihood = self.get_action_loglikelihood(self.actor_output, self.actions)
        ratio = tf.exp(action_loglikelihood-self.action_loglikelihood)
        pg_loss = tf.minimum(ratio*self.advantages, tf.clip_by_value(ratio, 0.8, 1.2)*self.advantages)
        self.actor_loss = -tf.reduce_mean(pg_loss)

    def build_loss(self):
        self.build_critic_loss()
        self.build_actor_loss()        
        