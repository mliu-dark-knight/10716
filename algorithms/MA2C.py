from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
from algorithms.common import *
from algorithms.A2C import Actor_, VNetwork, BetaActor
from utils import append_summary
import collections

class QNetwork(object):
    def __init__(self, hidden_dims, action_dims, scope):
        self.hidden_dims = hidden_dims
        self.action_dims = action_dims
        self.scope = scope
        
    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            for hidden_dim in self.hidden_dims:
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.variance_scaling())
            hidden = tf.layers.dense(hidden, reduce(mul, self.action_dims), activation=None,
                                   kernel_initializer=tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3))
            return tf.reshape(hidden, [-1]+self.action_dims)

class MA2C(object):
    '''Multi-agent Regret Policy Gradient.'''
    def __init__(self, env, hidden_dims,  gamma=1.0,
                 actor_lr=1e-3, critic_lr=1e-3, N=5):
        self.env = env
        self.hidden_dims = hidden_dims
        self.action_dim = []
        self.state_dim = []
        self.n_action = 0
        self.n_agent = len(self.env.observation_space)
        for i in range(self.n_agent):
            self.state_dim.append(reduce(mul, self.env.observation_space[i].shape))
            self.action_dim.append(self.env.action_space[i].n)
        self.n_action = sum(self.action_dim)
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.N = N
        self.actor_loss_list = []
        self.actor_list = []
        self.actor_step_list = []
        self.critic_loss_list = []
        self.critic_list = []
        self.critic_step_list = []
        self.alpha_list = []
        self.beta_list = []
        self.build()
        
    def build(self):
        self.build_actor_critic()
        self.build_placeholder()
        self.build_loss()
        self.build_step()
        self.build_summary()
    
    def build_actor_critic(self):
        for i in range(self.n_agent):
            self.actor_list.append(BetaActor(self.hidden_dims, self.action_dim[i], 'actor_{}'.format(i)))
            #self.critic_list.append(QNetwork(self.hidden_dims, self.action_dim, 'critic_{}'.format(i)))
            self.critic_list.append(VNetwork(self.hidden_dims, 'critic_{}'.format(i)))

    def build_placeholder(self):
        self.states = tf.placeholder(tf.float32, shape=[None, sum(self.state_dim)])
        self.actions = tf.placeholder(tf.float32, shape=[None, sum(self.action_dim)])
        self.rewards = tf.placeholder(tf.float32, shape=[None, self.n_agent])
        self.nexts = tf.placeholder(tf.float32, shape=[None, sum(self.state_dim)])
        self.are_non_terminal = tf.placeholder(tf.float32, shape=[None, self.n_agent])
        self.training = tf.placeholder(tf.bool)

    def aggregate_policy_for_value(self, states, agent_id):
        critic_output = self.critic_list[agent_id](states)
        policy_list = []
        for agent in range(self.n_agent):
            actor_states = states[:, sum(self.state_dim[:agent]):sum(self.state_dim[:agent+1])]
            pi = tf.nn.softmax(self.actor_list[agent](actor_states), axis=-1)
            policy_list.append(pi)
        joint_policy = policy_list[0][:,:, None]
        symbols = 'bcdefghijk'
        for agent in range(1, self.n_agent):
            policy = policy_list[agent][:, None, :]
            einsum_eq = 'a'+symbols[:agent+1]+','+'a'+symbols[agent]+symbols[agent+1]+'->a'+symbols[:agent]+symbols[agent+1]
            joint_policy = tf.einsum(einsum_eq, joint_policy, policy)
            joint_policy = tf.expand_dims(joint_policy, axis=agent+2)
        joint_policy = tf.squeeze(joint_policy, axis=-1)
        value = critic_output*joint_policy
        value = tf.reduce_sum(value, axis=[i for i in range(1, self.n_agent+1)])
        return value[:, None]
    
    def aggregate_action_for_qvalue(self, states, agent_id):
        batch_size = tf.shape(states)[0]
        batch_indices = tf.range(batch_size, dtype=tf.int32)[:, None] 
        critic_output = self.critic_list[agent_id](states)
        qvalues = []
        for action in range(self.action_dim[agent_id]):
            indices = [batch_indices]
            for agent in range(self.n_agent):
                if agent == agent_id:
                    indices.append(action*tf.ones(batch_size, dtype=tf.int32)[:, None])
                else:
                    indices.append(self.actions[:, agent_id][:, None])
            indices = tf.concat(indices, axis=1)
            qvalue = tf.gather_nd(critic_output, indices)
        qvalues.append(qvalue[:, None])
        return tf.concat(qvalues, axis=1)

    '''def build_loss(self):
        states = self.states
        nexts = self.nexts
        rewards = self.rewards
        batch_size = tf.shape(states)[0]
        batch_indices = tf.range(batch_size, dtype=tf.int32)[:, None]
        action_indices = [self.actions[:, agent_id][:, None] for agent_id in range(self.n_agent)]

        for agent_id in range(self.n_agent):
            next_value = tf.stop_gradient(self.aggregate_policy_for_value(nexts, agent_id))
            target_qvalue = tf.stop_gradient(rewards[:, agent_id][:, None]+self.are_non_terminal[:, agent_id][:, None] *\
                np.power(self.gamma, self.N)*next_value)
            critic_output = self.critic_list[agent_id](states)
            qvalue = tf.gather_nd(critic_output, tf.concat([batch_indices]+action_indices, axis=1))[:, None]
            #critic_loss = tf.losses.mean_squared_error(qvalue, target_qvalue)
            critic_loss = tf.losses.huber_loss(qvalue, target_qvalue)
            self.critic_loss_list.append(critic_loss)
            
            agent_states = self.states[:, sum(self.state_dim[:agent_id]):sum(self.state_dim[:agent_id+1])]
            action_indices_ = tf.concat([batch_indices, self.actions[:, agent_id][:, None]], axis=1)
            logits = self.actor_list[agent_id](agent_states)
            pi = tf.nn.softmax(logits, axis=-1)
            self.pi_list.append(pi)            
            qvalue = tf.stop_gradient(self.aggregate_action_for_qvalue(states, agent_id))
            value = tf.reduce_sum(qvalue*pi, axis=1)[:, None]
            #regret = tf.reduce_sum(tf.maximum(qvalue-value, 0), axis=1)
            regret = tf.math.softplus(tf.gather_nd(qvalue, action_indices_)[:, None]-value)
            if agent_id == 0:
                self.regret = qvalue-value
            log_pi = tf.log(pi+1e-12)
            log_action_prob = tf.gather_nd(log_pi, action_indices)[:, None]
            pg_loss = tf.reduce_mean(log_action_prob*regret)
            self.actor_loss_list.append(pg_loss)'''

    def build_loss(self):
        states = self.states
        nexts = self.nexts
        rewards = self.rewards
        batch_size = tf.shape(states)[0]
        batch_indices = tf.range(batch_size, dtype=tf.int32)[:, None]

        for agent_id in range(self.n_agent):
            target_value = tf.stop_gradient(rewards[:, agent_id][:, None]+self.are_non_terminal[:, agent_id][:, None] *\
                np.power(self.gamma, self.N)*self.critic_list[agent_id](nexts))
            value = self.critic_list[agent_id](states)
            #critic_loss = tf.losses.mean_squared_error(qvalue, target_qvalue)
            critic_loss = tf.losses.huber_loss(target_value, value)
            self.critic_loss_list.append(critic_loss)
            
            agent_states = self.states[:, sum(self.state_dim[:agent_id]):sum(self.state_dim[:agent_id+1])]
            agent_actions = self.actions[:, sum(self.action_dim[:agent_id]):sum(self.action_dim[:agent_id+1])]
            alpha, beta = self.actor_list[agent_id](agent_states)
            self.alpha_list.append(alpha)
            self.beta_list.append(beta)
         
            advantage = target_value-value
            #regret = tf.math.softplus(tf.gather_nd(qvalue, action_indices_)[:, None]-value)
            e = tf.ones(batch_size)*1e-12
            log_action_prob = (alpha-1.)*tf.log(agent_actions+e[:, None])+(beta-1.)*tf.log(1-agent_actions+e[:, None])
            normalization = -tf.math.lgamma(alpha)-tf.math.lgamma(beta)+tf.math.lgamma(alpha+beta)
            log_action_prob += normalization
            pg_loss = -tf.reduce_mean(tf.reduce_sum(log_action_prob, axis=1)*advantage)
            self.actor_loss_list.append(pg_loss)

    def build_step(self):
        self.global_step = tf.Variable(0, trainable=False)
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.critic_loss_list[0]), [self.critic_loss_list[0]])]):
            for i in range(self.n_agent):
                critic_step = critic_optimizer.minimize(
                self.critic_loss_list[i],
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_{}'.format(i)),
                global_step=self.global_step)
                self.critic_step_list.append(critic_step)

        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.actor_loss_list[0]), [self.actor_loss_list[0]])]):
            for i in range(self.n_agent):
                actor_step = actor_optimizer.minimize(
                self.actor_loss_list[i],
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_{}'.format(i)))
                self.actor_step_list.append(actor_step)
        
    
    def build_summary(self):
        for agent in range(self.n_agent):
            tf.summary.scalar('critic_loss_{}'.format(agent), self.critic_loss_list[agent])
            #tf.summary.scalar('actor_loss_{}'.format(agent), self.actor_loss_list[agent])
        self.merged_summary_op = tf.summary.merge_all()
    
    def train(self, sess, saver, summary_writer, progress_fd, model_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, max_episode_len=25, **kargs):
        total_rewards = []
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_meta = tf.RunMetadata()
        for i_episode in tqdm(range(train_episodes), ncols=100):
            states, actions, returns, nexts, are_non_terminal, total_reward = self.collect_trajectory(max_episode_len)
            feed_dict={self.states: states,
                                    self.actions: actions,
                                    self.rewards: returns,
                                    self.nexts: nexts,
                                    self.are_non_terminal: are_non_terminal,
                                    self.training: True}
            for t in range(step):
                sess.run(self.critic_step_list, feed_dict=feed_dict)    
            sess.run(self.actor_step_list, feed_dict=feed_dict)
            critic_loss = self.critic_loss_list[0].eval(feed_dict=feed_dict).mean()
            actor_loss = self.actor_loss_list[0].eval(feed_dict=feed_dict).mean()
            append_summary(progress_fd, str(start_episode + i_episode) + ",{0:.2f}".format(total_reward)\
                +",{0:.4f}".format(actor_loss)+",{0:.4f}".format(critic_loss))
            total_rewards.append(total_reward)
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
        return states[:-1], np.array(actions), returns, nexts, are_non_terminal, rewards[:,0].sum()
    
    def sample_action(self, state, agent_id):
        feed_dict = {self.states: state.reshape(1, -1), self.training: False}
        alpha = self.alpha_list[agent_id].eval(feed_dict=feed_dict).squeeze(axis=0)
        beta = self.beta_list[agent_id].eval(feed_dict=feed_dict).squeeze(axis=0)
        a = np.random.beta(alpha, beta)
        return a

    def generate_episode(self, render=False,
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
                a = self.sample_action(state, i)
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