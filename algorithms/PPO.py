from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from algorithms.common import *
from algorithms.A2C import A2C, Actor_, BetaActor, GaussianActor
from utils import append_summary
import gym

class PPO(A2C):
    '''
    Implement PPO algorithm.
    '''
    def __init__(self, *args, lambd=0.95, horrizon=2048,  **kwargs):
        self.lambd = lambd
        self.horrizon = horrizon
        self.env_info = {'done': True, 'last_state': None, 'total_reward': 0}
        super(PPO, self).__init__(*args, **kwargs)
        self.running_state = ZFilter((self.state_dim, ), clip=5)
        
        

    def build_placeholder(self):
        self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        if self.action_type == "continuous":
            self.actions = tf.placeholder(tf.float32, shape=[None, self.n_action])
        else:
            self.actions = tf.placeholder(tf.int32, shape=[None, 1])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.action_loglikelihood = tf.placeholder(tf.float32, shape=[None, 1])
        self.advantages = tf.placeholder(tf.float32, shape=[None, 1])
        self.returns = tf.placeholder(tf.float32, shape=[None, 1])
        self.nexts = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.are_non_terminal = tf.placeholder(tf.float32, shape=[None])
        self.training = tf.placeholder(tf.bool)
    
    def build_copy_op(self):
        pass
        '''self.init_actor, self.update_actor = get_target_updates(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'),
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor'), 1)'''

    def build_actor(self):
        if self.action_type == "discrete":
            self.actor = Actor_(self.hidden_dims, self.n_action, self.scope_pre+'actor')
        else:
            self.actor = GaussianActor(self.hidden_dims, self.n_action, self.scope_pre+'actor')

    def build_critic_loss(self):
        self.values = self.critic(self.states)
        self.critic_loss = 0.5*tf.losses.mean_squared_error(self.returns, self.values)
        

    def build_actor_loss(self):
        self.actor_output = self.actor(self.states)
        action_loglikelihood = self.get_action_loglikelihood(self.actor_output, self.actions)
        ratio = tf.exp(action_loglikelihood-self.action_loglikelihood)
        pg_loss = tf.minimum(ratio*self.advantages, tf.clip_by_value(ratio, 0.8, 1.2)*self.advantages)
        self.actor_loss = -tf.reduce_mean(pg_loss)

    def build_loss(self):
        self.build_critic_loss()
        self.build_actor_loss()
        

    def collect_transitions(self, sess):
        total_rewards = []
        states = np.zeros((self.horrizon+1, self.state_dim))
        if self.action_type == "continuous":
            actions = np.zeros((self.horrizon, self.n_action))
        else:
            actions = np.zeros((self.horrizon, 1))
        rewards = np.zeros(self.horrizon)
        are_non_terminal = np.zeros(self.horrizon)
        action_loglikelihood = np.zeros((self.horrizon, 1))
        for step in range(self.horrizon):
            if self.env_info['done']:
                if self.is_env_pool:
                    self.env = self.env_pool.sample_env()
                state = self.env.reset()
                state = self.running_state(state)
                self.env_info['last_state'] = state
                self.env_info['done'] = False
                self.env_info['total_reward'] = 0
            state = self.env_info['last_state']
            action, loglikelihood = self.sample_action(state)
            states[step]=state
            state, reward, done, _ = self.env.step(action)
            state = self.running_state(state)
            rewards[step]=reward
            self.env_info['last_state'] = state
            self.env_info['total_reward'] += reward
            actions[step,:] = action
            action_loglikelihood[step, :] = loglikelihood
            if done:
                are_non_terminal[step] = 0
                self.env_info['done'] = True
                total_rewards.append(self.env_info['total_reward'])
                self.env_info['total_reward'] = 0
            else:
                are_non_terminal[step] = 1
        step += 1
        states[step]=state
        if isinstance(self.actor, BetaActor):
            actions = self.convert_action_for_beta_actor(actions)
        values = sess.run(self.values, feed_dict={self.states: states})
        returns = np.zeros((self.horrizon, 1))
        deltas = np.zeros((self.horrizon, 1))
        advantages = np.zeros((self.horrizon, 1))

        prev_return = values[-1]
        prev_value = values[-1]
        values = values[:-1]
        prev_advantage = 0
        for i in reversed(range(self.horrizon)):
            returns[i] = rewards[i] + self.gamma * prev_return * are_non_terminal[i]
            deltas[i] = rewards[i] + self.gamma * prev_value * are_non_terminal[i] - values[i]
            advantages[i] = deltas[i] + self.gamma * self.lambd * prev_advantage * are_non_terminal[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        if len(total_rewards)>0:
            avg_rewards = np.mean(total_rewards)
        else:
            avg_rewards = None
        return states[:-1], actions, action_loglikelihood, returns, advantages, avg_rewards
    
    def generate_episode(self, render=False):
        states = []
        actions = []
        rewards = []
        if self.is_env_pool:
            self.env = self.env_pool.sample_env()
        state = self.env.reset()
        while True:
            action, _ = self.sample_action(state)
            states.append(state)
            state, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()
            if self.action_type == "discrete":
                actions.append([action])
            else:
                actions.append(action)
            rewards.append(reward)
            if done:
                break
        states.append(state)

        actions = np.array(actions)
        if isinstance(self.actor, BetaActor):
            actions = self.convert_action_for_beta_actor(actions)
        assert len(states)  == len(actions)+1 and \
               len(actions) == len(rewards)
        return states, actions, rewards

    '''def build_step(self):
        def clip_grad_by_global_norm(grad_var, max_norm):
            grad_var = list(zip(*grad_var))
            grad, var = grad_var[0], grad_var[1]
            clipped_grad,_ = tf.clip_by_global_norm(grad, max_norm)
            return list(zip(clipped_grad, var))

        self.global_step = tf.Variable(0, trainable=False)
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr, epsilon=1e-5)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.actor_loss), [self.actor_loss])]):
            gvs = actor_optimizer.compute_gradients(self.actor_loss,
               var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'))
            clipped_grad_var = clip_grad_by_global_norm(gvs, 1)
            self.actor_step = actor_optimizer.apply_gradients(clipped_grad_var)
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr, epsilon=1e-5)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.critic_loss), [self.critic_loss])]):
            gvs = critic_optimizer.compute_gradients(self.critic_loss,
               var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))
            #clipped_grad_var = clip_grad_by_global_norm(gvs, 1)
            self.critic_step = actor_optimizer.apply_gradients(clipped_grad_var)'''

    def build_step(self):
        self.global_step = tf.Variable(0., trainable=False)
        actor_lr = tf.train.polynomial_decay(self.actor_lr, self.global_step, 6000*self.horrizon, 3e-5)

        actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr, epsilon=1e-5)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.actor_loss), [self.actor_loss])]):
            self.actor_step = actor_optimizer.minimize(
                self.actor_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'))

        critic_lr = tf.train.polynomial_decay(self.critic_lr, self.global_step, 6000, 2e-4)
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr, epsilon=1e-5)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.critic_loss), [self.critic_loss])]):
            self.critic_step = critic_optimizer.minimize(
                self.critic_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'),
                global_step=self.global_step)
    
    def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, epsilon=0.3, apply_her=False, n_goals=10, train_steps=-1):
        total_rewards = []
        n_step = 0
        i_episode = 0
        for i_episode in tqdm(range(train_episodes), ncols=100):
            states_mem, actions_mem, action_loglikelihood_mem, returns_mem, advantage_mem, epi_avg_reward = self.collect_transitions(sess)
            #self.global_step.assign_add(1)
            for s in range(step):
                perm = np.random.permutation(len(states_mem))
                for sample_id in range(0, len(perm), batch_size):
                    feed_dict = {self.states: states_mem[perm[sample_id: sample_id+batch_size]],
                                 self.actions: actions_mem[perm[sample_id: sample_id+batch_size]],
                                 self.action_loglikelihood: action_loglikelihood_mem[perm[sample_id: sample_id+batch_size]],
                                 self.returns: returns_mem[perm[sample_id: sample_id+batch_size]],
                                 self.advantages: advantage_mem[perm[sample_id: sample_id+batch_size]],
                                 self.training: True}
                    sess.run([self.actor_step, self.critic_step], feed_dict=feed_dict)
            n_step += len(states_mem)
            if not epi_avg_reward is None:
                append_summary(progress_fd, str(start_episode+i_episode) + ",{0:.2f}".format(epi_avg_reward)+ ",{}".format(n_step))
                total_rewards.append(epi_avg_reward)
            if (i_episode + 1) % save_episodes == 0:
                saver.save(sess, model_path)
        return total_rewards