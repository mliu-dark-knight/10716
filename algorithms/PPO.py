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
    Implement off-policy A2C algorithm.
    Advantage is estimated as V(next)-V(current)+reward.
    '''
    def __init__(self, *args, lambd=0.95, horrizon=2048, **kwargs):
        self.lambd = lambd
        self.horrizon = horrizon
        self.env_info = {'done': True, 'last_state': None, 'total_reward': 0}
        super(PPO, self).__init__(*args, **kwargs)
        self.build_copy_op()
        

    def build_placeholder(self):
        self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        if self.action_type == "continuous":
            self.actions = tf.placeholder(tf.float32, shape=[None, self.n_action])
        else:
            self.actions = tf.placeholder(tf.int32, shape=[None, 1])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.advantages = tf.placeholder(tf.float32, shape=[None, 1])
        self.returns = tf.placeholder(tf.float32, shape=[None, 1])
        self.nexts = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.are_non_terminal = tf.placeholder(tf.float32, shape=[None])
        self.training = tf.placeholder(tf.bool)
    
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

    def compute_gae(self, target_value, value):
        td_err = target_value - value
        batch_size = len(value)
        if batch_size == 1:
            return td_err
        a = np.ones(batch_size-1)*self.lambd*self.gamma
        a = np.expand_dims(np.cumprod(a), axis=0)
        a = np.concatenate([np.ones((1,1)), a], axis=1)

        b = np.ones(batch_size-1)/(self.lambd*self.gamma)
        b = np.expand_dims(np.cumprod(b), axis=1)
        b = np.concatenate([np.ones((1,1)), b], axis=0)
        mask = np.dot(b,a)
        mask = np.triu(mask)
        advantage = np.dot(mask, td_err)
        return advantage

    def compute_return(self, rewards):
        batch_size = len(rewards)
        rewards = rewards.reshape((-1, 1))
        if batch_size == 1:
            return rewards
        a = np.ones(batch_size-1)*self.gamma
        a = np.expand_dims(np.cumprod(a), axis=0)
        a = np.concatenate([np.ones((1,1)), a], axis=1)

        b = np.ones(batch_size-1)/(self.gamma)
        b = np.expand_dims(np.cumprod(b), axis=1)
        b = np.concatenate([np.ones((1,1)), b], axis=0)
        mask = np.dot(b,a)
        mask = np.triu(mask)
        returns  = np.dot(mask, rewards)
        return returns

    def build_critic_loss(self):
        self.critic_loss = tf.losses.mean_squared_error(self.returns, self.value)
        #self.critic_loss += tf.losses.get_regularization_loss(scope=self.scope_pre+"critic")
        

    def build_actor_loss(self):
        self.actor_output = self.actor(self.states)
        action_loglikelihood = self.get_action_loglikelihood(self.actor_output, self.actions)
        old_action_loglikelihood = self.get_action_loglikelihood(self.actor_target(self.states), self.actions)
        ratio = tf.exp(action_loglikelihood-old_action_loglikelihood)
        pg_loss = tf.minimum(ratio*self.advantages, tf.clip_by_value(ratio, 0.8, 1.2)*self.advantages)
        #self.actor_loss = -tf.reduce_mean(pg_loss)
        #entropy = self.get_policy_entropy(self.actor_output)
        #self.actor_loss = -tf.reduce_mean(pg_loss+1e-2*entropy)
        self.actor_loss = -tf.reduce_mean(pg_loss)

    def build_loss(self):
        self.value = self.critic(self.states)
        self.target_value = tf.stop_gradient(self.rewards[:, None]+self.are_non_terminal[:, None] * \
                   np.power(self.gamma, self.N) * self.critic(self.nexts))
        self.build_actor_loss()
        self.build_critic_loss()

    def collect_transitions(self, sess):
        states_mem = None
        returns_mem = None
        advantages_mem = None
        action_meme = None
        total_rewards = []
        states = []
        actions = []
        rewards = []

        for step in range(self.horrizon):
            if self.env_info['done']:
                state = self.env.reset()
                self.env_info['last_state'] = state
                self.env_info['done'] = False
                self.env_info['total_reward'] = 0
            state = self.env_info['last_state']
            action = self.sample_action(state)
            states.append(state)
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            self.env_info['last_state'] = state
            self.env_info['total_reward'] += reward
            if self.action_type == "discrete":
                actions.append([action])
            else:
                actions.append(action)

            if done:
                states.append(state)
                self.env_info['done'] = True
                total_rewards.append(self.env_info['total_reward'])
                self.env_info['total_reward'] = 0

                actions = np.array(actions)
                if isinstance(self.actor, BetaActor):
                    actions += (self.action_upper_limit + self.action_lower_limit)/2.
                    actions /= self.action_upper_limit - self.action_lower_limit
                states = np.array(states)
                rewards = np.array(rewards)
                nexts = states[1:].copy()
                are_non_terminal = np.ones(len(nexts))
                are_non_terminal[-1] = 0
                total_rewards.append(sum(rewards))
                feed_dict = {self.states: states[:-1], self.rewards: rewards,
                            self.nexts: nexts, self.are_non_terminal: are_non_terminal, self.training: True}
                target_value, value = sess.run([self.target_value, self.value], feed_dict=feed_dict)
                advantages = self.compute_gae(target_value, value)
                returns = self.compute_return(rewards)
                if states_mem is None:
                    states_mem = states[:-1]
                    returns_mem = returns
                    advantages_mem = advantages
                    actions_mem = actions
                else:
                    states_mem = np.concatenate([states_mem, states[:-1]], axis=0)
                    returns_mem = np.concatenate([returns_mem, returns], axis=0)
                    advantages_mem = np.concatenate([advantages_mem, advantages], axis=0)
                    actions_mem = np.concatenate([actions_mem, actions], axis=0)
                states = []
                actions = []
                rewards = []
        if not self.env_info['done']:
            states.append(state)
            actions = np.array(actions)
            if isinstance(self.actor, BetaActor):
                actions += (self.action_upper_limit + self.action_lower_limit)/2.
                actions /= self.action_upper_limit - self.action_lower_limit
            states = np.array(states)
            rewards = np.array(rewards)
            nexts = states[1:].copy()
            are_non_terminal = np.ones(len(nexts))
            feed_dict = {self.states: states[:-1], self.rewards: rewards,
                        self.nexts: nexts, self.are_non_terminal: are_non_terminal}
            target_value, value = sess.run([self.target_value, self.value], feed_dict=feed_dict)
            advantages = self.compute_gae(target_value, value)
            returns = self.compute_return(rewards)
            states_mem = np.concatenate([states_mem, states[:-1]], axis=0)
            returns_mem = np.concatenate([returns_mem, returns], axis=0)
            advantages_mem = np.concatenate([advantages_mem, advantages], axis=0)
            actions_mem = np.concatenate([actions_mem, actions], axis=0)

        #return states[:-1], actions, returns, nexts, are_non_terminal, total_reward
        assert len(states_mem)  == len(actions_mem) and \
               len(actions_mem) == len(returns_mem)
        #advantages_mem = (advantages_mem - advantages_mem.mean())/advantages_mem.std()
        return states_mem, actions_mem, returns_mem, advantages_mem, total_rewards
    
    def generate_episode(self, render=False):
        states = []
        actions = []
        rewards = []
        state = self.env.reset()
        while True:
            action = self.sample_action(state)
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
            actions += (self.action_upper_limit + self.action_lower_limit)/2.
            actions /= self.action_upper_limit - self.action_lower_limit
        assert len(states)  == len(actions)+1 and \
               len(actions) == len(rewards)
        return states, actions, rewards
    
    def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, epsilon=0.3, apply_her=False, n_goals=10, train_steps=-1):
        total_rewards = []
        sess.run([self.init_actor])
        n_step = 0
        i_episode = 0
        if train_episodes > 0:
            raise NotImplementedError
        else:
            pbar = tqdm(total=train_steps)

            while n_step < train_steps:
                #states, actions, returns, nexts, are_non_terminal, total_reward = self.collect_trajectory()
                #feed_dict = {self.states: states, self.actions: actions, self.rewards: returns,
                #            self.nexts: nexts, self.are_non_terminal: are_non_terminal, self.training: True}
                states_mem, actions_mem, returns_mem, advantage_mem, total_reward = self.collect_transitions(sess)
                
                for s in range(step):
                    perm = np.random.permutation(len(states_mem))
                    for sample_id in range(0, len(perm), batch_size):
                        feed_dict = {self.states: states_mem[sample_id: sample_id+batch_size],
                                     self.actions: actions_mem[sample_id: sample_id+batch_size],
                                     self.returns: returns_mem[sample_id: sample_id+batch_size],
                                     self.advantages: advantage_mem[sample_id: sample_id+batch_size],
                                     self.training: True}
                        sess.run([self.actor_step, self.critic_step], feed_dict=feed_dict)
                sess.run([self.update_actor])
                n_step += len(states_mem)
                for reward in total_reward:
                    i_episode += 1
                    append_summary(progress_fd, str(start_episode+i_episode) + ",{0:.2f}".format(reward))
                if (i_episode + 1) % save_episodes == 0:
                    saver.save(sess, model_path)
                total_rewards+=total_reward
                pbar.update(len(states_mem))
        return total_rewards