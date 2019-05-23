from functools import reduce
from operator import mul
from .QRDQN import QRQNetwork
from typing import *
from tqdm import tqdm
import numpy as np
from algorithms.common import *
from utils import append_summary

class CentralizedQRQNetwork(object):
    def __init__(self, action_dim, hidden_dims, n_quantile, scope):
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.n_quantile = n_quantile
        self.scope = scope
    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.variance_scaling())
            hidden = tf.layers.dense(hidden, self.n_quantile*reduce(mul, self.action_dim), activation=None,
                                   kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3))
            return tf.reshape(hidden, [-1, reduce(mul, self.action_dim), self.n_quantile])


class MultiSoftQRDDPG(object):
    def __init__(self, env, hidden_dims, temperature=1, kappa=1.0, n_quantile=64, replay_memory=None, gamma=1.0, tau=1e-2,
                 actor_lr=1e-3, critic_lr=1e-3, N=5):
        self.env = env
        self.temperature = temperature
        self.hidden_dims = hidden_dims
        self.action_dim = []
        self.state_dim = []
        self.n_action = 0
        self.n_agent = len(self.env.observation_space)
        for i in range(self.n_agent):
            self.state_dim.append(reduce(mul, self.env.observation_space[i].shape))
            self.action_dim.append(self.env.action_space[i].n)
        self.n_action = sum(self.action_dim)
        self.kappa = kappa
        self.n_quantile = n_quantile
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.N = N
        self.replay_memory = replay_memory
        self.tau = tau
        self.actor_loss_list = []
        self.actor_list = []
        self.actor_step_list = []
        self.critic_loss_list = []
        self.critic_list = []
        self.critic_target_list = []
        self.init_critic_list =[]
        self.update_critic_list =[]
        self.critic_step_list = []
        self.logits_list = []
        self.build()
        
    def build(self):
        self.build_actor_critic()
        self.build_placeholder()
        self.build_loss()
        self.build_copy_op()
        self.build_step()
        self.build_summary()
    
    def build_actor_critic(self):
        for i in range(self.n_agent):
            self.actor_list.append(Actor(self.hidden_dims, self.action_dim[i], 'actor_{}'.format(i)))
            self.critic_list.append(CentralizedQRQNetwork(self.action_dim, self.hidden_dims, self.n_quantile, 'critic_{}'.format(i)))
            self.critic_target_list.append(CentralizedQRQNetwork(self.action_dim, self.hidden_dims, self.n_quantile, 'critic_target_{}'.format(i)))

    def build_placeholder(self):
        self.states = tf.placeholder(tf.float32, shape=[None, sum(self.state_dim)], name="states")
        self.actions = tf.placeholder(tf.int32, shape=[None,self.n_agent], name="actions")
        self.rewards = tf.placeholder(tf.float32, shape=[None, self.n_agent], name="rewards")
        self.nexts = tf.placeholder(tf.float32, shape=[None, sum(self.state_dim)], name="nexts")
        self.are_non_terminal = tf.placeholder(tf.float32, shape=[None, self.n_agent], name="are_non_terminal")
        self.training = tf.placeholder(tf.bool)
    
    def build_copy_op(self):
        for i in range(self.n_agent):
            init_critic, update_critic = get_target_updates(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_{}'.format(i)),
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_target_{}'.format(i)), self.tau)
            self.update_critic_list.append(update_critic)
            self.init_critic_list.append(init_critic)

    def action_encoding(self, actions):
        base = [1]
        for i in range(1, self.n_agent):
            base.append(int(base[-1]*self.action_dim[i-1]))
        base = base[::-1]
        
        encoded = tf.cast(actions[0]*base[0], dtype=tf.int32)
        for j in range(1, self.n_agent):
            encoded += tf.cast(actions[j]*base[j], dtype=tf.int32)
        return encoded

    def action_decoding(self, encoded, agent_id):
        base = [1]
        for i in range(1, self.n_agent):
            base.append(base[-1]*self.action_dim[i-1])
        base = base[::-1]
        actions = []
        for j in range(self.n_agent):
            action = tf.cast(tf.floor(encoded / base[j]), dtype=tf.int32)
            encoded = encoded % base[j]
            actions.append(action)

    def build_loss(self):
        with tf.variable_scope('normalize_states'):
            bn = tf.layers.BatchNormalization(_reuse=tf.AUTO_REUSE)
            states = bn.apply(self.states, training=self.training)
            nexts = bn.apply(self.nexts, training=self.training)
        batch_size = tf.shape(states)[0]
        batch_indices = tf.range(batch_size, dtype=tf.int32)[:, None]
        for i in range(self.n_agent):
            target_network_output = self.critic_target_list[i](nexts)
            target_action_qvalue_all = tf.reduce_mean(target_network_output, axis=-1)
            action_list = [self.actions[:,j] for j in range(self.n_agent)]
            action_qvalue_list = []
            for action_id in range(self.action_dim[i]):
                this_action = tf.zeros(batch_size)+action_id
                action_list[i] = this_action
                encoded = self.action_encoding(action_list)[:, None]
                indices = tf.concat([batch_indices, encoded], axis=1)
                action_qvalue = tf.gather_nd(target_action_qvalue_all,
                                indices)[:, None]
                action_qvalue_list.append(action_qvalue)
            action_qvalue = tf.concat(action_qvalue_list, axis=1)
            target_action = tf.cast(tf.argmax(action_qvalue, axis=1)[:, None],
                                                                dtype=tf.int32)
            target_action_indices = tf.concat([batch_indices, target_action],
                                               axis=1)          
            target_quantiles = tf.gather_nd(target_network_output,
                                        target_action_indices)
            target_Z = self.rewards[:,i][:, None]+self.are_non_terminal[:, i][:, None] * \
                   np.power(self.gamma, self.N) * target_quantiles            
            network_output = self.critic_list[i](states)
            action_list = [self.actions[:,j] for j in range(self.n_agent)]
            encoded = self.action_encoding(action_list)[:, None]
            action_indices = tf.concat([batch_indices, encoded], axis=1)
            Z = tf.gather_nd(network_output, action_indices)
            qvalue = tf.reduce_mean(Z, axis=-1)
            
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
            
            agent_states = states[:, sum(self.state_dim[:i]):sum(self.state_dim[:i+1])]
            logits = tf.nn.log_softmax(self.actor_list[i](agent_states))
            self.logits_list.append(logits-tf.log(-tf.log(tf.random_uniform(tf.shape(logits)))))
            log_pi = tf.nn.log_softmax(self.logits_list[i], axis=-1)
            action_indices = tf.concat([batch_indices, self.actions[:, i][:, None]], axis=1)
            self.actor_loss_list.append(-tf.reduce_mean(qvalue*tf.gather_nd(log_pi, action_indices)))

    def build_step(self):
        self.global_step = tf.Variable(0, trainable=False)
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.critic_loss_list[0]), [self.critic_loss_list[0]])]):
            for i in range(self.n_agent):
                actor_step = actor_optimizer.minimize(
                self.actor_loss_list[i],
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_{}'.format(i)))
                self.actor_step_list.append(actor_step)
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='normalize_states') +
                                     [tf.Assert(tf.is_finite(self.critic_loss_list[0]), [self.critic_loss_list[0]])]):
            for i in range(self.n_agent):
                critic_step = critic_optimizer.minimize(
                self.critic_loss_list[i],
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_{}'.format(i)),
                global_step=self.global_step)
                self.critic_step_list.append(critic_step)
    
    def build_summary(self):
        tf.summary.scalar('critic_loss', self.critic_loss_list[0])
        self.merged_summary_op = tf.summary.merge_all()
    
    def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, max_episode_len=25, **kargs):
        total_rewards = []
        sess.run(self.init_critic_list)
        for i_episode in tqdm(range(train_episodes), ncols=100):
            total_reward = self.collect_trajectory(max_episode_len)
            append_summary(progress_fd, str(start_episode + i_episode) + ',{0:.2f}'.format(total_reward))
            total_rewards.append(total_reward)
            states, actions, rewards, nexts, are_non_terminal = self.replay_memory.sample_batch(step * batch_size)
            for t in range(step):
                sess.run(self.critic_step_list,
                         feed_dict={self.states: states[t * batch_size: (t + 1) * batch_size],
                                    self.actions: actions[t * batch_size: (t + 1) * batch_size],
                                    self.rewards: rewards[t * batch_size: (t + 1) * batch_size],
                                    self.nexts: nexts[t * batch_size: (t + 1) * batch_size],
                                    self.are_non_terminal: are_non_terminal[t * batch_size: (t + 1) * batch_size],
                                    self.training: True})
                
                sess.run(self.actor_step_list,
                         feed_dict={self.states: states[t * batch_size: (t + 1) * batch_size],
                                    self.actions: actions[t * batch_size: (t + 1) * batch_size],
                                    self.nexts: nexts[t * batch_size: (t + 1) * batch_size],
                                    self.rewards: rewards[t * batch_size: (t + 1) * batch_size],
                                    self.are_non_terminal: are_non_terminal[t * batch_size: (t + 1) * batch_size],
                                    self.training: True})
                sess.run(self.update_critic_list)
            # summary_writer.add_summary(summary, global_step=self.global_step.eval())
            if (i_episode + 1) % save_episodes == 0:
                saver.save(sess, model_path)
        return total_rewards
    
    def normalize_returns(self, nexts: np.array, rewards: List) -> Tuple[List, np.array, List]:
        '''
        Compute N step returns
        :param states:
        :param rewards:
        :return:
        WARNING: make sure states, rewards come from the same episode
        '''
        T = len(nexts)
        assert len(rewards) == T
        nexts = np.roll(nexts, -self.N, axis=0)
        are_non_terminal = np.ones(T)
        if T >= self.N:
            are_non_terminal[T - self.N:] = 0
        else:
            are_non_terminal[:] = 0

        returns = []
        for reward in reversed(rewards):
            return_ = reward if len(returns) == 0 else reward + self.gamma * returns[-1]
            returns.append(return_)
        returns.reverse()
        returns = np.array(returns)
        if T > self.N:
            returns -= np.power(self.gamma, self.N) \
                       * np.pad(returns[self.N:], (0, self.N), 'constant', constant_values=(0))

        return returns, nexts, are_non_terminal
        
    def collect_trajectory(self, max_episode_len):
        states, actions, rewards = self.generate_episode(max_episode_len=max_episode_len)
        states = np.array(states)
        rewards = np.array(rewards)
        returns = []
        nexts = []
        are_non_terminal = []
        for i in range(self.n_agent):
            returns_, nexts, are_non_terminal_ = self.normalize_returns(states[1:,], rewards[:,i])
            returns.append(returns_)
            are_non_terminal.append(are_non_terminal_)
        returns = np.array(returns).T
        nexts = np.array(nexts)
        are_non_terminal = np.array(are_non_terminal).T
        self.replay_memory.append(states[:-1],
                                  actions, returns,
                                  nexts,
                                  are_non_terminal)

        return rewards[:,0].sum()
    
    def generate_episode(self, render=False,
                         max_episode_len=25, benchmark=False):
        states = []
        actions = []
        rewards = []
        state = self.env.reset()
        state = np.concatenate([j.reshape(1,-1) for j in state], axis=1).flatten()
        step = 0
        while True:
            states.append(state)
            action = []
            for i in range(self.n_agent):
                logits = self.logits_list[i].eval(feed_dict={
                self.states: np.expand_dims(state, axis=0),
                self.training: False})
                logits = logits.squeeze(axis=0)
                pi = np.exp(logits)
                pi /= pi.sum()
                action.append(np.random.choice(np.arange(self.action_dim[i]), size=None, replace=True, p=pi))
            state, reward, done, info = self.env.step(action)
            state = np.concatenate([j.reshape(1,-1) for j in state], axis=1).flatten()
            if render:
                self.env.render()
            actions.append(action)
            rewards.append(reward)
            step += 1
            if all(done) or step >= max_episode_len:
                #print("Break. Step={}".format(step))
                break
        states.append(state)
        assert len(states)  == len(actions)+1 and \
               len(actions) == len(rewards)
        if not benchmark:
            return states, actions, rewards
        else:
            return states, actions, rewards, info