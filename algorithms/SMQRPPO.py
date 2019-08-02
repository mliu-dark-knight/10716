from typing import *
import numpy as np
import tensorflow as tf
from algorithms.common import *
from algorithms.MQRPPO import MQRPPO
from utils import append_summary
import gym

class SMQRPPO(MQRPPO):
    def __init__(self, *args, en_coe=0.01, **kwargs):
        self.en_coe = 0.2
        super(SMQRPPO, self).__init__(*args, **kwargs)

    def build_actor_loss(self):
        for agent_id in range(self.n_agent):
            agent_states = self.states[:, sum(self.state_dim[:agent_id]):sum(self.state_dim[:agent_id+1])]
            agent_actions = self.actions[:, sum(self.action_dim[:agent_id]):sum(self.action_dim[:agent_id+1])]
            actor_output = self.actor_list[agent_id](agent_states)
            self.actor_output_list.append(actor_output)
            action_loglikelihood = self.get_agent_action_loglikelihood(agent_actions, actor_output)

            ratio = tf.exp(action_loglikelihood-self.action_loglikelihood[:, agent_id][:, None])
            adv = self.advantages[:, agent_id][:, None] + self.en_coe*(self.action_loglikelihood-action_loglikelihood)
            pg_loss = tf.minimum(ratio*adv, tf.clip_by_value(ratio, 0.8, 1.2)*adv)
            actor_loss = -tf.reduce_mean(pg_loss)+tf.losses.get_regularization_loss(scope="actor_{}".format(agent_id))
            self.actor_loss_list.append(actor_loss)
    def collect_transitions(self, sess, max_episode_len=25):
        total_rewards = []
        states = np.zeros((self.horrizon+1, sum(self.state_dim)))
        actions = np.zeros((self.horrizon, sum(self.action_dim)))
        rewards = np.zeros((self.horrizon, self.n_agent))
        are_non_terminal = np.zeros((self.horrizon, self.n_agent))
        actions_loglikelihood = np.zeros((self.horrizon, self.n_agent))
        episode_steps = 0
        for step in range(self.horrizon):
            if self.env_info['done']:
                state = self.env.reset()
                episode_steps = 0
                state = np.concatenate([j.reshape(1,-1) for j in state], axis=1).flatten()
                for agent_id in range(self.n_agent):
                    obs_start = sum(self.state_dim[:agent_id])
                    obs_end = sum(self.state_dim[:agent_id+1])
                    #state[obs_start:obs_end] = self.running_state_list[agent_id](state[obs_start:obs_end])
                self.env_info['last_state'] = state
                self.env_info['done'] = False
                self.env_info['total_reward'] = 0
            state = self.env_info['last_state']
            states[step] = state
            
            action = []
            loglikelihood = []
            for i in range(self.n_agent):
                a, ll = self.sample_action(sess, state, i)
                action.append(a)
                loglikelihood.append(ll)
            state, reward, done, _ = self.env.step(action)
            episode_steps += 1
            state = np.concatenate([j.reshape(1,-1) for j in state], axis=1).flatten()
            for agent_id in range(self.n_agent):
                obs_start = sum(self.state_dim[:agent_id])
                obs_end = sum(self.state_dim[:agent_id+1])
                #state[obs_start:obs_end] = self.running_state_list[agent_id](state[obs_start:obs_end])
            self.env_info['last_state'] = state
            self.env_info['total_reward'] += reward[0]
            rewards[step]=reward-self.en_coe*np.array(loglikelihood)
            action = np.concatenate([j.reshape(1, -1) for j in action], axis=1).flatten()
            loglikelihood = np.concatenate([j.reshape(1, -1) for j in loglikelihood], axis=1).flatten()
            actions[step,:] = action
            actions_loglikelihood[step, :] = loglikelihood
            if all(done) or episode_steps >= max_episode_len:
                are_non_terminal[step, :] = np.zeros(self.n_agent)
                self.env_info['done'] = True
                total_rewards.append(self.env_info['total_reward'])
                self.env_info['total_reward'] = 0
            else:
                are_non_terminal[step, :] = np.ones(self.n_agent)
        step += 1
        states[step]=state
        returns = np.zeros((self.horrizon, self.n_agent))
        deltas = np.zeros((self.horrizon, self.n_agent))
        advantages = np.zeros((self.horrizon, self.n_agent))
        values = sess.run(self.values_list, feed_dict={self.states: states})
        for agent_id in range(self.n_agent):
            agent_values = values[agent_id]
            prev_return = agent_values[-1]
            prev_value = agent_values[-1]
            agent_values = agent_values[:-1]
            prev_advantage = 0
            for i in reversed(range(self.horrizon)):
                returns[i, agent_id] = rewards[i, agent_id] + self.gamma * prev_return * are_non_terminal[i, agent_id]
                deltas[i, agent_id] = rewards[i, agent_id] + self.gamma * prev_value * are_non_terminal[i, agent_id] - agent_values[i]
                advantages[i, agent_id] = deltas[i, agent_id] + self.gamma * self.lambd * prev_advantage * are_non_terminal[i, agent_id]
                prev_return = returns[i, agent_id]
                prev_value = agent_values[i]
                prev_advantage = advantages[i, agent_id]
            advantages[:, agent_id] = (advantages[:, agent_id] - advantages[:, agent_id].mean()) / (advantages[:, agent_id].std() + 1e-9)
        return states[:-1], actions, actions_loglikelihood, returns, advantages, np.mean(total_rewards)
