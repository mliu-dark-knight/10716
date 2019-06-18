from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from algorithms.common import *
from algorithms.A2C import A2C, Actor_, BetaActor
from utils import append_summary
import gym

class PPO(A2C):
    '''
    Implement off-policy A2C algorithm.
    Advantage is estimated as V(next)-V(current)+reward.
    '''
    def __init__(self, *args, lambd=0.95, **kwargs):
        self.lambd = lambd
        super(PPO, self).__init__(*args, **kwargs)
        self.build_copy_op()
    
    def build_copy_op(self):
        self.init_actor, self.update_actor = get_target_updates(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'),
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor'), 1)

    def build_actor(self):
        if self.action_type == "discrete":
            self.actor = Actor_(self.hidden_dims, self.n_action, self.scope_pre+'actor')
            self.actor_target = Actor_(self.hidden_dims, self.n_action, self.scope_pre+'target_actor')
        else:
            self.actor = BetaActor(self.hidden_dims, self.n_action, self.scope_pre+'actor')
            self.actor_target = BetaActor(self.hidden_dims, self.n_action, self.scope_pre+'target_actor')

    def compute_gae(self, value, target_value):
        td_err = target_value - value
        batch_size = tf.shape(value)[0]
        a = tf.ones(batch_size-1)*self.lambd*self.gamma
        a = tf.math.cumprod(a, axis=0)[None, :]
        a = tf.concat([tf.ones((1,1)), a], axis=1)

        b = tf.ones(batch_size-1)/(self.lambd*self.gamma)
        b = tf.math.cumprod(b, axis=0)[:, None]
        b = tf.concat([tf.ones((1,1)), b], axis=0)
        mask = tf.linalg.matmul(b,a)
        mask = tf.matrix_band_part(mask, 0, -1)
        advantage = tf.linalg.matmul(mask, td_err)
        return advantage

    def compute_return(self):
        batch_size = tf.shape(self.states)[0]
        a = tf.ones(batch_size-1)*self.gamma
        a = tf.math.cumprod(a, axis=0)[None, :]
        a = tf.concat([tf.ones((1,1)), a], axis=1)
        b = tf.ones(batch_size-1)/self.gamma
        b = tf.math.cumprod(b, axis=0)[:, None]
        b = tf.concat([tf.ones((1,1)), b], axis=0)
        mask = tf.linalg.matmul(b,a)
        mask = tf.matrix_band_part(mask, 0, -1)
        self.returns = tf.linalg.matmul(mask, self.rewards[:, None])

    def build_critic_loss(self):
        self.compute_return()
        self.value = self.critic(self.states)
        self.critic_loss = tf.losses.mean_squared_error(self.returns, self.value)
        #self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic")
        self.target_value = tf.stop_gradient(self.rewards[:,None]+self.are_non_terminal[:, None] * \
                   np.power(self.gamma, self.N) * self.critic(self.nexts))

    def build_loss(self):
        self.build_critic_loss()
        batch_size = tf.shape(self.states)[0]
        batch_indices = tf.range(batch_size,dtype=tf.int32)[:, None]
        self.actor_output = self.actor(self.states)
        action_loglikelihood = self.get_action_loglikelihood(self.actor_output, self.actions)
        old_action_loglikelihood = self.get_action_loglikelihood(self.actor_target(self.states), self.actions)
        ratio = tf.exp(action_loglikelihood-old_action_loglikelihood)
        advantage = tf.stop_gradient(self.compute_gae(self.value, self.target_value))
        pg_loss = tf.minimum(ratio*advantage, tf.clip_by_value(ratio, 0.8, 1.2)*advantage)
        #self.actor_loss = -tf.reduce_mean(pg_loss)
        entropy = self.get_policy_entropy(self.actor_output)
        self.actor_loss = -tf.reduce_mean(pg_loss+1e-2*entropy)

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
            #clipped_grad_var = clip_grad_by_global_norm(gvs, 0.5)
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
            #gvs = critic_optimizer.compute_gradients(self.critic_loss,
            #    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))
            #clipped_grad_var = clip_grad_by_global_norm(gvs, 0.5)
            #self.critic_step = actor_optimizer.apply_gradients(clipped_grad_var)
    
    def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, epsilon=0.3, apply_her=False, n_goals=10, train_steps=-1):
        total_rewards = []
        sess.run([self.init_actor])
        n_step = 0
        if train_episodes > 0:
            for i_episode in tqdm(range(train_episodes), ncols=100):
                states, actions, returns, nexts, are_non_terminal, total_reward = self.collect_trajectory()
                feed_dict = {self.states: states, self.actions: actions, self.rewards: returns,
                            self.nexts: nexts, self.are_non_terminal: are_non_terminal, self.training: True}
                total_rewards.append(total_reward)
                perm = np.random.permutation(len(states))
                for s in range(step):
                    sess.run([self.actor_step], feed_dict=feed_dict)
                sess.run([self.update_actor])
                for s in range(3*step):
                    sess.run([self.critic_step], feed_dict=feed_dict)
                critic_loss = self.critic_loss.eval(feed_dict=feed_dict).mean()
                actor_loss = self.actor_loss.eval(feed_dict=feed_dict).mean()
                n_step += len(states)
                append_summary(progress_fd, str(start_episode + i_episode) + ",{0:.2f}".format(total_reward)\
                    +",{0:.8f}".format(actor_loss)+",{0:.4f}".format(critic_loss)+",{}".format(n_step))
                if (i_episode + 1) % save_episodes == 0:
                    saver.save(sess, model_path)
        else:
            pbar = tqdm(total=train_steps)
            i_episode = 0
            while n_step < train_steps:
                states, actions, returns, nexts, are_non_terminal, total_reward = self.collect_trajectory()
                feed_dict = {self.states: states, self.actions: actions, self.rewards: returns,
                            self.nexts: nexts, self.are_non_terminal: are_non_terminal, self.training: True}
                total_rewards.append(total_reward)
                perm = np.random.permutation(len(states))
                for s in range(step):
                    sess.run([self.actor_step], feed_dict=feed_dict)
                sess.run([self.update_actor])
                for s in range(3*step):
                    sess.run([self.critic_step], feed_dict=feed_dict)
                critic_loss = self.critic_loss.eval(feed_dict=feed_dict).mean()
                actor_loss = self.actor_loss.eval(feed_dict=feed_dict).mean()
                n_step += len(states)
                i_episode += 1
                append_summary(progress_fd, str(start_episode + i_episode) + ",{0:.2f}".format(total_reward)\
                    +",{0:.8f}".format(actor_loss)+",{0:.4f}".format(critic_loss)+",{}".format(n_step))
                if (i_episode + 1) % save_episodes == 0:
                    saver.save(sess, model_path)
                pbar.update(len(states))
        return total_rewards