from typing import *
import numpy as np
import tensorflow as tf
from algorithms.common import *
from algorithms.MPPO import MPPO
from algorithms.QRA2C import QRVNetwork, QRVNetworkNoCrossing
from utils import append_summary
import gym

class MQRPPO(MPPO):
    def __init__(self, *args, kappa=1.0, n_quantile=200, **kwargs):
        self.kappa = kappa
        self.n_quantile = n_quantile
        super(MQRPPO, self).__init__(*args, **kwargs)
    
    def get_mean(self, Z):
        part1 = Z[:, :-2:2]
        part2 = Z[:,1:-1:2]
        part3 = Z[:, 2::2]
        Z = (part1+4.*part2+part3)/3.
        return tf.reduce_mean(Z, axis=1)

    def build_critic(self):
        for i in range(self.n_agent):
            self.critic_list.append(QRVNetworkNoCrossing(self.hidden_dims, self.n_quantile, 'critic_{}'.format(i)))
    
    def build_critic_loss(self):
        for agent_id in range(self.n_agent):
            agent_states = self.states[:, sum(self.state_dim[:agent_id]):sum(self.state_dim[:agent_id+1])]
            Z = self.critic_list[agent_id](agent_states)
            #values = self.get_mean(Z)[:, None]
            values = Z[:, int(self.n_quantile/2)]
            self.values_list.append(values)

            errors = self.returns[:, agent_id][:, None] - Z
            huber_loss_case_one = tf.to_float(tf.abs(errors) <= self.kappa) * 0.5 * errors ** 2
            huber_loss_case_two = tf.to_float(tf.abs(errors) > self.kappa) * self.kappa * \
                                    (tf.abs(errors) - 0.5 * self.kappa)
            huber_loss = huber_loss_case_one + huber_loss_case_two
            quantiles = tf.expand_dims(tf.linspace(1. / self.n_quantile, 1-1. / self.n_quantile, self.n_quantile), axis=0)
            quantile_huber_loss = (tf.abs(quantiles - tf.stop_gradient(tf.to_float(errors < 0))) * huber_loss) / \
                              self.kappa
            critic_loss = tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=1), axis=0)
            #critic_loss += tf.losses.get_regularization_loss(scope="critic_{}".format(agent_id))
            self.critic_loss_list.append(critic_loss)
