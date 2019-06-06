from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
from algorithms.common import *
from algorithms.A2C import BetaActor
from algorithms.MA2C import MA2C
from algorithms.QRA2C import QRVNetwork
from utils import append_summary
import collections

class MQRA2C(MA2C):
    '''Multi-agent version of QRA2C.State is defined to be concantenation of four consecutive observation of all agents.'''
    def __init__(self, *args, kappa=1.0, n_quantile=64, **kwargs):
        self.kappa = kappa
        self.n_quantile = n_quantile
        super(MQRA2C, self).__init__(*args, **kwargs)
    
    def build_actor_critic(self):
        for i in range(self.n_agent):
            self.actor_list.append(BetaActor(self.hidden_dims, self.action_dim[i], 'actor_{}'.format(i)))
            self.critic_list.append(QRVNetwork(self.hidden_dims, self.n_quantile, 'critic_{}'.format(i)))

    def get_mean(self, Z):
        part1 = Z[:, :-2:2]
        part2 = Z[:,1:-1:2]
        part3 = Z[:, 2::2]
        Z = (part1+4.*part2+part3)/3.
        return tf.reduce_mean(Z, axis=1)

    def build_loss(self):
        states = self.states
        nexts = self.nexts
        rewards = 0.01*self.rewards
        batch_size = tf.shape(states)[0]
        batch_indices = tf.range(batch_size, dtype=tf.int32)[:, None]

        for agent_id in range(self.n_agent):
            target_Z = tf.stop_gradient(rewards[:, agent_id][:, None]+self.are_non_terminal[:, agent_id][:, None] * \
                   np.power(self.gamma, self.N) * self.critic_list[agent_id](nexts))          
            Z = self.critic_list[agent_id](states)
            bellman_errors = tf.expand_dims(target_Z, axis=2) - tf.expand_dims(Z, axis=1)
            huber_loss_case_one = tf.to_float(tf.abs(bellman_errors) <= self.kappa) * 0.5 * bellman_errors ** 2
            huber_loss_case_two = tf.to_float(tf.abs(bellman_errors) > self.kappa) * self.kappa * \
                              (tf.abs(bellman_errors) - 0.5 * self.kappa)
            huber_loss = huber_loss_case_one + huber_loss_case_two
            quantiles = tf.expand_dims(tf.expand_dims(tf.range(0.5 / self.n_quantile, 1., 1. / self.n_quantile), axis=0),
                                   axis=1)
            quantile_huber_loss = (tf.abs(quantiles - tf.stop_gradient(tf.to_float(bellman_errors < 0))) * huber_loss) / \
                              self.kappa
            self.critic_loss_list.append(tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=2), axis=1), axis=0))
            
            agent_states = self.states[:, sum(self.state_dim[:agent_id]):sum(self.state_dim[:agent_id+1])]
            agent_actions = self.actions[:, sum(self.action_dim[:agent_id]):sum(self.action_dim[:agent_id+1])]
            alpha, beta = self.actor_list[agent_id](agent_states)
            self.alpha_list.append(alpha)
            self.beta_list.append(beta)
            advantage = tf.stop_gradient(self.get_mean(target_Z)-self.get_mean(Z))
            e = tf.ones(batch_size)*1e-12
            log_action_prob = (alpha-1.)*tf.log(agent_actions+e[:, None])+(beta-1.)*tf.log(1-agent_actions+e[:, None])
            normalization = -tf.math.lgamma(alpha)-tf.math.lgamma(beta)+tf.math.lgamma(alpha+beta)
            log_action_prob += normalization
            pg_loss = -tf.reduce_mean(tf.reduce_sum(log_action_prob, axis=1)*advantage)
            self.actor_loss_list.append(pg_loss)