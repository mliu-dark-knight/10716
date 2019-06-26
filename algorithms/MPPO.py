from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
from algorithms.common import *
from algorithms.A2C import Actor_, VNetwork, BetaActor, GaussianActor
from utils import append_summary
import collections
from scipy.special import loggamma 

class MPPO(object):
    '''Multi-agent Proximal Policy Gradient.'''
    def __init__(self, env, hidden_dims,  gamma=0.99,
                 actor_lr=1e-4, critic_lr=1e-4, lambd=0.95, horrizon=2048):
        self.env = env
        self.env_info = {'done': True, 'last_state': None, 'total_reward': 0}
        self.hidden_dims = hidden_dims
        self.action_dim = []
        self.state_dim = []
        self.n_agent = len(self.env.observation_space)
        self.running_state_list = []
        for i in range(self.n_agent):
            self.state_dim.append(reduce(mul, self.env.observation_space[i].shape))
            self.action_dim.append(self.env.action_space[i].n)
            self.running_state_list.append(ZFilter((self.state_dim[i], ), clip=5))
        self.gamma = gamma
        self.lambd = lambd
        self.horrizon = horrizon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_loss_list = []
        self.actor_list = []
        self.actor_step_list = []
        self.critic_loss_list = []
        self.critic_list = []
        self.critic_step_list = []
        self.values_list = []
        self.actor_output_list = []
        self.build()
    
    def save_state_filter(self, path):
        shape = self.running_state_list[0].rs._M.shape[0]
        nms = np.zeros((self.n_agent, 3, shape))
        for agent_id in range(self.n_agent):
            nms[agent_id, 0, 0] = self.running_state_list[agent_id].rs._n
            nms[agent_id, 1] = self.running_state_list[agent_id].rs._M
            nms[agent_id, 2] = self.running_state_list[agent_id].rs._S
        np.save(path, nms)

    def load_state_filter(self, path):
        nms = np.load(path)
        for agent_id in range(self.n_agent):
            self.running_state_list[agent_id].rs._n = nms[agent_id, 0, 0]
            self.running_state_list[agent_id].rs._M = nms[agent_id, 1]
            self.running_state_list[agent_id].rs._S = nms[agent_id, 2]

    def build(self):
        self.build_actor()
        self.build_critic()
        self.build_placeholder()
        self.build_loss()
        self.build_step()
        self.build_summary()
    
    def build_actor(self):
        for i in range(self.n_agent):
            self.actor_list.append(GaussianActor(self.hidden_dims, self.action_dim[i], 'actor_{}'.format(i)))
    def build_critic(self):
        for i in range(self.n_agent):
            self.critic_list.append(VNetwork(self.hidden_dims, 'critic_{}'.format(i)))

    def build_placeholder(self):
        self.states = tf.placeholder(tf.float32, shape=[None, sum(self.state_dim)])
        self.actions = tf.placeholder(tf.float32, shape=[None, sum(self.action_dim)])
        self.rewards = tf.placeholder(tf.float32, shape=[None, self.n_agent])
        self.are_non_terminal = tf.placeholder(tf.float32, shape=[None, self.n_agent])
        self.training = tf.placeholder(tf.bool)
        self.action_loglikelihood = tf.placeholder(tf.float32, shape=[None, self.n_agent])
        self.advantages = tf.placeholder(tf.float32, shape=[None, self.n_agent])
        self.returns = tf.placeholder(tf.float32, shape=[None, self.n_agent])

    def build_loss(self):
        self.build_critic_loss()
        self.build_actor_loss()
    
    def build_critic_loss(self):
        for agent_id in range(self.n_agent):
            agent_states = self.states[:, sum(self.state_dim[:agent_id]):sum(self.state_dim[:agent_id+1])]
            values = self.critic_list[agent_id](agent_states)
            self.values_list.append(values)
            self.critic_loss_list.append(0.5*tf.losses.mean_squared_error(self.returns[:, agent_id][:, None],self.values_list[agent_id]))
    
    def get_agent_action_loglikelihood(self, agent_actions, actor_output):
        if isinstance(self.actor_list[0], BetaActor):
            alpha, beta = actor_output[0], actor_output[1]
            action_loglikelihood = (alpha-1.)*tf.log(agent_actions+1e-10)+(beta-1.)*tf.log(1-agent_actions+1e-10)
            action_loglikelihood += -tf.math.lgamma(alpha)-tf.math.lgamma(beta)+tf.math.lgamma(alpha+beta)
            action_loglikelihood = tf.reduce_sum(action_loglikelihood, axis=1)[:, None]
            return action_loglikelihood
        elif isinstance(self.actor_list[0], GaussianActor):
            mean, log_std = actor_output[0], actor_output[1]
            std = tf.exp(log_std)
            action_loglikelihood = -0.5*tf.log(2*np.pi)-log_std
            action_loglikelihood += -0.5*tf.pow((agent_actions-mean)/std, 2)
            action_loglikelihood = tf.reduce_sum(action_loglikelihood, axis=1)[:, None]
            return action_loglikelihood
        elif isinstance(self.actor_list[0], DirichletActor):
            alpha = actor_output
            action_loglikelihood = -tf.math.lbeta(alpha)
            action_loglikelihood += tf.reduce_sum((alpha-1)*tf.log(agent_actions+1e-10), axis=1)[:, None]
            return action_loglikelihood
        else:
            raise NotImplementedError

    def build_actor_loss(self):
        for agent_id in range(self.n_agent):
            agent_states = self.states[:, sum(self.state_dim[:agent_id]):sum(self.state_dim[:agent_id+1])]
            agent_actions = self.actions[:, sum(self.action_dim[:agent_id]):sum(self.action_dim[:agent_id+1])]
            actor_output = self.actor_list[agent_id](agent_states)
            self.actor_output_list.append(actor_output)
            action_loglikelihood = self.get_agent_action_loglikelihood(agent_actions, actor_output)

            ratio = tf.exp(action_loglikelihood-self.action_loglikelihood[:, agent_id][:, None])
            adv = self.advantages[:, agent_id][:, None]
            pg_loss = tf.minimum(ratio*adv, tf.clip_by_value(ratio, 0.8, 1.2)*adv)
            self.actor_loss_list.append(-tf.reduce_mean(pg_loss))

    def build_step(self):
        self.global_step = tf.Variable(0, trainable=False)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.critic_loss_list[0]), [self.critic_loss_list[0]])]):
            for i in range(self.n_agent):
                agent_critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr, epsilon=1e-5)
                critic_step = agent_critic_optimizer.minimize(
                self.critic_loss_list[i],
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_{}'.format(i)),
                global_step=self.global_step)
                self.critic_step_list.append(critic_step)
        
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.actor_loss_list[0]), [self.actor_loss_list[0]])]):
            for i in range(self.n_agent):
                agent_actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr, epsilon=1e-5)
                actor_step = agent_actor_optimizer.minimize(
                self.actor_loss_list[i],
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_{}'.format(i)))
                self.actor_step_list.append(actor_step)
        
    def build_summary(self):
        for agent in range(self.n_agent):
            tf.summary.scalar('critic_loss_{}'.format(agent), self.critic_loss_list[agent])
            #tf.summary.scalar('actor_loss_{}'.format(agent), self.actor_loss_list[agent])
        self.merged_summary_op = tf.summary.merge_all()

    def train(self, sess, saver, summary_writer, progress_fd, model_path, filter_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, max_episode_len=25, **kargs):
        total_rewards = []
        n_step = 0
        i_episode = 0
        for i_episode in tqdm(range(train_episodes), ncols=100):
            states_mem, actions_mem, action_loglikelihood_mem, returns_mem, advantage_mem, epi_avg_reward = self.collect_transitions(sess, max_episode_len)
            for s in range(step):
                perm = np.random.permutation(len(states_mem))
                for sample_id in range(0, len(perm), batch_size):
                    feed_dict = {self.states: states_mem[perm[sample_id: sample_id+batch_size]],
                                 self.actions: actions_mem[perm[sample_id: sample_id+batch_size]],
                                 self.action_loglikelihood: action_loglikelihood_mem[perm[sample_id: sample_id+batch_size]],
                                 self.returns: returns_mem[perm[sample_id: sample_id+batch_size]],
                                 self.advantages: advantage_mem[perm[sample_id: sample_id+batch_size]],
                                 self.training: True}
                    #sess.run(self.actor_step_list+self.critic_step_list, feed_dict=feed_dict)
                    sess.run(self.actor_step_list, feed_dict=feed_dict)
                    for j in range(5):
                        sess.run(self.critic_step_list, feed_dict=feed_dict)
            n_step += len(states_mem)
            append_summary(progress_fd, str(start_episode+i_episode) + ",{0:.2f}".format(epi_avg_reward)+ ",{}".format(n_step))
            total_rewards.append(epi_avg_reward)
            if (i_episode + 1) % save_episodes == 0:
                saver.save(sess, model_path)
                self.save_state_filter(filter_path)
        return total_rewards

    def sample_action(self, sess, states, agent_id):
        feed_dict = {self.states: states.reshape(1, -1), self.training: False}
        if isinstance(self.actor_list[0], BetaActor):
            alpha = self.actor_output_list[agent_id][0].eval(feed_dict=feed_dict).squeeze(axis=0)
            beta = self.actor_output_list[agent_id][1].eval(feed_dict=feed_dict).squeeze(axis=0)
            action = np.random.beta(alpha, beta)
            action_loglikelihood = (alpha-1.)*np.log(action+1e-10)+(beta-1.)*np.log(1-action+1e-10)
            action_loglikelihood += -loggamma(alpha)-loggamma(beta)+loggamma(alpha+beta)
            action_loglikelihood = np.sum(action_loglikelihood)
        elif isinstance(self.actor_list[0], GaussianActor):
            mean, log_std = sess.run(self.actor_output_list[agent_id], feed_dict=feed_dict)
            mean = mean.squeeze(axis=0)
            log_std = log_std.squeeze(axis=0)
            std = np.exp(log_std)
            action = np.random.normal(loc=mean, scale=std)
            action_loglikelihood = -0.5*np.log(2*np.pi)-log_std
            action_loglikelihood += -0.5*np.power((action-mean)/std, 2)
            action_loglikelihood = np.sum(action_loglikelihood)
        elif isinstance(self.actor_list[0], DirichletActor):
            alpha = self.actor_output_list[agent_id].eval(feed_dict=feed_dict).squeeze(axis=0)
            action = np.random.dirichlet(alpha)
            action_loglikelihood = -np.sum(loggamma(alpha))+loggamma(np.sum(alpha))
            action_loglikelihood += np.sum((alpha-1)*np.log(action))
        return action, action_loglikelihood

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
                    state[obs_start:obs_end] = self.running_state_list[agent_id](state[obs_start:obs_end])
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
                state[obs_start:obs_end] = self.running_state_list[agent_id](state[obs_start:obs_end])
            self.env_info['last_state'] = state
            self.env_info['total_reward'] += reward[0]
            rewards[step]=reward
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
    
    def generate_episode(self, sess, render=False,
                         max_episode_len=25, benchmark=False):
        states = []
        actions = []
        rewards = []
        infos = []
        state = self.env.reset()
        state = np.concatenate([j.reshape(1,-1) for j in state], axis=1).flatten()
        step = 0
        while True:  
            action = []
            for i in range(self.n_agent):
                a, _ = self.sample_action(sess, state, i)
                action.append(a)
            states.append(state)
            state, reward, done, info = self.env.step(action)
            state = np.concatenate([j.reshape(1,-1) for j in state], axis=1).flatten()
            action = np.concatenate([j.reshape(1, -1) for j in action], axis=1).flatten()
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
            if render:
                self.env.render()
            step += 1
            if all(done) or step >= max_episode_len:
                break
        states.append(state)

        assert len(states)  == len(actions)+1 and \
               len(actions) == len(rewards)
        if not benchmark:
            return states, actions, rewards
        else:
            return states, actions, rewards, info