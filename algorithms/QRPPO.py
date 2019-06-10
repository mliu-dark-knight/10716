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
        self.critic_target = QRVNetworkNoCrossing(self.hidden_dims, self.n_quantile, self.scope_pre+'target_critic')

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
        self.target_Z = tf.stop_gradient(rewards[:,None]+self.are_non_terminal[:, None] * \
                   np.power(self.gamma, self.N) * self.critic_target(nexts))
        self.Z = self.critic(states)
        Z = self.Z
        target_Z = self.target_Z
        
        regression_loss = self.get_huber_quantile_regression_loss(target_Z, Z)
        self.critic_loss = tf.reduce_mean(self.get_huber_quantile_regression_loss(target_Z, Z), axis=0)
        self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic")

        target_value = self.get_mean(target_Z)
        value = self.get_mean(Z)
        #self.critic_loss = tf.losses.huber_loss(target_value, value)
        self.actor_output = self.actor(states)
        action_loglikelihood = self.get_action_loglikelihood(self.actor_output, self.actions)
        old_action_loglikelihood = self.get_action_loglikelihood(self.actor_target(states), self.actions)
        ratio = tf.exp(action_loglikelihood-old_action_loglikelihood)
        #advantage = tf.stop_gradient(target_value-value)
        advantage = tf.stop_gradient(self.compute_gae(value[:, None], target_value[:, None]))
        pg_loss = tf.minimum(ratio*advantage, tf.clip_by_value(ratio, 0.8, 1.2)*advantage)
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
            gvs = critic_optimizer.compute_gradients(self.critic_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))
            clipped_grad_var = clip_grad_by_global_norm(gvs, 0.5)
            self.critic_step = actor_optimizer.apply_gradients(clipped_grad_var)

    def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, epsilon=0.3, apply_her=False, n_goals=10):
        total_rewards = []
        quantile_outputs = []
        sess.run([self.init_actor, self.init_critic])
        for i_episode in tqdm(range(train_episodes), ncols=100):
            states, actions, returns, nexts, are_non_terminal, total_reward = self.collect_trajectory()
            feed_dict = {self.states: states[-2].reshape(1,-1), self.rewards: [returns[-2]],
                         self.nexts: nexts[-2].reshape(1,-1), self.are_non_terminal: [are_non_terminal[-2]], self.training: False}

            Z = self.Z.eval(feed_dict=feed_dict).squeeze()
            target_Z = self.target_Z.eval(feed_dict=feed_dict).squeeze()
            critic_loss = self.critic_loss.eval(feed_dict=feed_dict).squeeze()
            quantile_outputs.append([Z, target_Z, [critic_loss]])
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
        np.save("quantile_outputs.npy", np.array(quantile_outputs))
        return total_rewards