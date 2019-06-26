from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from algorithms.common import *
from algorithms.MPPO import MPPO
from algorithms.A2C import Actor_, BetaActor, GaussianActor
from algorithms.QRA2C import QRVNetwork, QRVNetworkNoCrossing
from utils import append_summary
import gym

class MSQRPPO(MPPO):
    def __init__(self, *args, kappa=1.0, quantile=0.5, **kwargs):
        self.kappa = kappa
        self.n_quantile = 1
        self.quantile = quantile
        super(MSQRPPO, self).__init__(*args, **kwargs)
    
    def build_critic(self):
        for i in range(self.n_agent):
            self.critic_list.append(QRVNetworkNoCrossing(self.hidden_dims, self.n_quantile, 'critic_{}'.format(i)))
    
    def build_critic_loss(self):
        for agent_id in range(self.n_agent):
            agent_states = self.states[:, sum(self.state_dim[:agent_id]):sum(self.state_dim[:agent_id+1])]
            values = self.critic_list[agent_id](agent_states)
            self.values_list.append(values)
            errors = self.returns - values
            huber_loss_case_one = tf.to_float(tf.abs(errors) <= self.kappa) * 0.5 * errors ** 2
            huber_loss_case_two = tf.to_float(tf.abs(errors) > self.kappa) * self.kappa * \
                                    (tf.abs(errors) - 0.5 * self.kappa)
            huber_loss = huber_loss_case_one + huber_loss_case_two
            quantile_huber_loss = (tf.abs(self.quantile - tf.stop_gradient(tf.to_float(errors < 0))) * huber_loss) / \
                                    self.kappa
            critic_loss = tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=1), axis=0)
            critic_loss += tf.losses.get_regularization_loss(scope="critic_{}".format(agent_id))
            self.critic_loss_list.append(critic_loss)
