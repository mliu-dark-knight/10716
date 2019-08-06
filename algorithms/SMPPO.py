from functools import reduce
from operator import mul
from typing import *
from tqdm import tqdm
import numpy as np
from algorithms.common import *
from utils import append_summary
import collections

class GaussianActor(object):
    def __init__(self, hidden_dims, action_dim, scope, policy_reg, n_agent):
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.scope = scope
        self.policy_reg = policy_reg
        self.n_agent = n_agent
        with tf.variable_scope(scope):
            self.log_std = tf.get_variable(name=self.scope+"_log_std",
                                           shape=[1, self.action_dim, self.n_agent],
                                           initializer=tf.zeros_initializer(),
                                           regularizer=tf.contrib.layers.l2_regularizer(self.policy_reg))

    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            hidden = tf.layers.dense(hidden, self.hidden_dims[0], activation=tf.nn.tanh,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(self.policy_reg),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(self.policy_reg),
                                         kernel_initializer=tf.initializers.orthogonal())
            mean = []
            for agent in range(self.n_agent):
                out = tf.layers.dense(hidden, self.hidden_dims[1], activation=tf.nn.tanh,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(self.policy_reg),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(self.policy_reg),
                                         kernel_initializer=tf.initializers.orthogonal())
                out = tf.layers.dense(out, self.action_dim, activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.policy_reg),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(self.policy_reg),
                                         kernel_initializer=tf.initializers.orthogonal(),
                                         use_bias=False)[:,:,None]
                mean.append(out)
            mean = tf.concat(mean, axis=2)
            batch_size = tf.shape(states)[0]
            log_std = tf.tile(self.log_std, (batch_size, 1,1))
            return tf.transpose(mean, perm=[0,2,1]), tf.transpose(log_std, perm=[0,2,1])

class QRVNetwork(object):
    def __init__(self, hidden_dims, scope, value_reg, n_agent):
        self.hidden_dims = hidden_dims
        self.scope = scope
        self.value_reg = value_reg
        self.n_agent = n_agent
        
    def __call__(self, states):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            hidden = states
            hidden = tf.layers.dense(hidden, self.hidden_dims[0], activation=tf.nn.tanh,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.value_reg),
                                     bias_regularizer=tf.contrib.layers.l2_regularizer(self.value_reg),
                                     kernel_initializer=tf.initializers.orthogonal())
            quantiles = []
            for agent in range(self.n_agent):
                quantile = tf.layers.dense(hidden, self.hidden_dims[1], activation=tf.nn.tanh,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.value_reg),
                                             bias_regularizer=tf.contrib.layers.l2_regularizer(self.value_reg),
                                             kernel_initializer=tf.initializers.orthogonal(),
                                             use_bias=False)
                quantile = tf.layers.dense(quantile, 1, activation=None,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(self.value_reg),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(self.value_reg),
                                         kernel_initializer=tf.initializers.orthogonal(),
                                         use_bias=False)[:,:, None]
                quantiles.append(quantile)
        quantiles = tf.concat(quantiles, axis=2)
        quantiles = tf.transpose(quantiles, perm=[0,2,1])
        return quantiles

class SMPPO(object):
    '''Multi-agent Proximal Policy Gradient.'''
    def __init__(self, env, hidden_dims,  gamma=0.99,
                 actor_lr=1e-4, critic_lr=1e-4,
                 lambd=0.96, horrizon=2048, policy_reg=0.,
                 value_reg=0.,quantile=0.5, kappa=1):
        self.env = env
        self.env_info = {'done': True, 'last_state': None, 'total_reward': 0}
        self.hidden_dims = hidden_dims
        self.n_agent = len(self.env.observation_space)
        self.state_dim=reduce(mul, self.env.observation_space[0].shape)
        self.action_dim=self.env.action_space[0].n
        self.running_state=ZFilter((self.state_dim, ), clip=40)
        self.gamma = gamma
        self.lambd = lambd
        self.quantile = quantile
        self.kappa = kappa
        self.horrizon = horrizon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_loss = None
        self.actor = None
        self.actor_step = None
        self.critic_loss = None
        self.critic = None
        self.critic_step = None
        self.values = None
        self.actor_output = None
        self.policy_reg = policy_reg
        self.value_reg = value_reg
        #self.tau = 0.01
        self.build()

    def build(self):
        self.build_actor()
        self.build_critic()
        self.build_placeholder()
        self.build_loss()
        self.build_step()
        #self.build_copy_op()
    
    def save_state_filter(self, path):
        return

    def load_state_filter(self, path):
        return

    def build_actor(self):
        self.actor = GaussianActor(self.hidden_dims, self.action_dim, 'actor', self.policy_reg, self.n_agent)
        #self.actor_target = GaussianActor(self.hidden_dims, self.action_dim, 'target_actor', self.policy_reg, self.n_agent)

    #def build_copy_op(self):
    #    self.init_actor, self.update_actor = get_target_updates(
    #        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'),
    #        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor'), self.tau)

    def build_critic(self):
        self.critic = QRVNetwork(self.hidden_dims, 'critic', self.value_reg, self.n_agent)

    def build_placeholder(self):
        self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.actions = tf.placeholder(tf.float32, shape=[None, self.action_dim])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.agent_id = tf.placeholder(tf.int32, shape=[None])
        self.action_loglikelihood = tf.placeholder(tf.float32, shape=[None])
        self.advantages = tf.placeholder(tf.float32, shape=[None])
        self.returns = tf.placeholder(tf.float32, shape=[None])

    def build_loss(self):
        self.build_critic_loss()
        self.build_actor_loss()
    
    def build_critic_loss(self):
        batch_size = tf.shape(self.states)[0]
        quantile_id = tf.concat([tf.range(batch_size, dtype=tf.int32)[:, None], self.agent_id[:, None]], axis=1)
        self.values = tf.gather_nd(self.critic(self.states), quantile_id)
        errors = self.returns[:, None] - self.values
        huber_loss_case_one = tf.to_float(tf.abs(errors) <= self.kappa) * 0.5 * errors ** 2
        huber_loss_case_two = tf.to_float(tf.abs(errors) > self.kappa) * self.kappa * \
                                (tf.abs(errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two
        quantile_huber_loss = (tf.abs(self.quantile - tf.stop_gradient(tf.to_float(errors < 0))) * huber_loss) / \
                                self.kappa
        critic_loss = tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=1), axis=0)
        critic_loss += tf.losses.get_regularization_loss(scope="critic")
        self.critic_loss = critic_loss


    def get_agent_action_loglikelihood(self, agent_actions, actor_output):
        mean, log_std = actor_output[0], actor_output[1]
        std = tf.exp(log_std)
        action_loglikelihood = -0.5*tf.log(2*np.pi)-log_std
        action_loglikelihood += -0.5*tf.pow((agent_actions-mean)/std, 2)
        #action_loglikelihood -= tf.log(1e-6+1-tf.tanh(agent_actions)**2)
        action_loglikelihood = tf.reduce_sum(action_loglikelihood, axis=1)[:, None]
        return action_loglikelihood

    def build_actor_loss(self):
        batch_size = tf.shape(self.states)[0]
        action_id = tf.concat([tf.range(batch_size,dtype=tf.int32)[:, None], self.agent_id[:, None]], axis=1)
        mean, log_std = self.actor(self.states)
        mean = tf.gather_nd(mean, action_id)
        log_std = tf.gather_nd(log_std, action_id)
        self.actor_output = (mean, log_std)

        #mean_target, log_std_target = self.actor_target(self.states)
        #mean_target = tf.gather_nd(mean_target, action_id)
        #log_std_target = tf.gather_nd(log_std_target, action_id)
        #self.actor_output_target = (mean_target, log_std_target)

        action_loglikelihood = self.get_agent_action_loglikelihood(self.actions, self.actor_output)
        ratio = tf.exp(action_loglikelihood-self.action_loglikelihood[:, None])
        adv = self.advantages[:, None]
        pg_loss = tf.minimum(ratio*adv, tf.clip_by_value(ratio, 0.8, 1.2)*adv)
        self.actor_loss = -tf.reduce_mean(pg_loss)+tf.losses.get_regularization_loss(scope="actor")

    def build_step(self):
        self.global_step = tf.Variable(0, trainable=False)
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.critic_loss), [self.critic_loss])]):
            critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr, epsilon=1e-5)
            self.critic_step = critic_optimizer.minimize(
            self.critic_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'),
            global_step=self.global_step)
        
        with tf.control_dependencies([tf.Assert(tf.is_finite(self.actor_loss), [self.actor_loss])]):
            agent_actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr, epsilon=1e-5)
            self.actor_step = agent_actor_optimizer.minimize(
            self.actor_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'))

    def train(self, sess, saver, summary_writer, progress_fd, model_path, filter_path, batch_size=64, step=10, start_episode=0,
              train_episodes=1000, save_episodes=100, max_episode_len=25, **kargs):
        total_rewards = []
        n_step = 0
        i_episode = 0
        #sess.run([self.init_actor])
        for i_episode in tqdm(range(train_episodes), ncols=100):
            states_mem, actions_mem, action_loglikelihood_mem, returns_mem, advantage_mem, agent_id_mem, epi_avg_reward = self.collect_transitions(sess, max_episode_len)
            for s in range(step):
                perm = np.random.permutation(len(states_mem))
                for sample_id in range(0, len(perm), batch_size):
                    feed_dict = {self.states: states_mem[perm[sample_id: sample_id+batch_size]],
                                 self.actions: actions_mem[perm[sample_id: sample_id+batch_size]],
                                 self.action_loglikelihood: action_loglikelihood_mem[perm[sample_id: sample_id+batch_size]],
                                 self.returns: returns_mem[perm[sample_id: sample_id+batch_size]],
                                 self.advantages: advantage_mem[perm[sample_id: sample_id+batch_size]],
                                 self.agent_id: agent_id_mem[perm[sample_id: sample_id+batch_size]]}
                    sess.run([self.actor_step,self.critic_step], feed_dict=feed_dict)
            #print(self.actor_loss.eval(feed_dict=feed_dict))
            #if i_episode == 10:
            #    exit()
            #sess.run([self.update_actor])
            n_step += self.horrizon
            append_summary(progress_fd, str(start_episode+i_episode) + ",{0:.2f}".format(epi_avg_reward)+ ",{}".format(n_step))
            total_rewards.append(epi_avg_reward)
            if (i_episode + 1) % save_episodes == 0:
                saver.save(sess, model_path)
        return total_rewards
    def get_deterministic_action(self, sess, states, agent_id):
        feed_dict = {self.states: states.reshape(1, -1), 
                     self.agent_id: np.array[agent_id]}
        mean, log_std = sess.run(self.actor_output, feed_dict=feed_dict)
        return mean.squeeze(axis=0)

    def sample_action(self, sess, states, agent_id):
        feed_dict = {self.states: states.reshape(1, -1),
                     self.agent_id: np.array([agent_id])}
        mean, log_std = sess.run(self.actor_output, feed_dict=feed_dict)
        mean = mean.squeeze(axis=0)
        log_std = log_std.squeeze(axis=0)
        std = np.exp(log_std)
        action = np.random.normal(loc=mean, scale=std)
        action_loglikelihood = -0.5*np.log(2*np.pi)-log_std
        action_loglikelihood += -0.5*np.power((action-mean)/std, 2)
        #action_loglikelihood -= np.log(1e-6+1-np.tanh(action)**2)
        action_loglikelihood = np.sum(action_loglikelihood)
        return action, action_loglikelihood

    def sample_training_action(self, sess, states):
        agent_to_train = np.random.randint(low=0, high=self.n_agent)
        action = []
        for i in range(self.n_agent):
            obs_start = i*self.state_dim
            obs_end = (i+1)*self.state_dim
            if i == agent_to_train:
                a, ll = self.sample_action(sess, states[obs_start: obs_end], i)
                action_to_train = a
                loglikelihood_to_train = ll
                state_to_train = states[obs_start: obs_end]
            else:
                feed_dict = {self.states: states[obs_start: obs_end].reshape(1, -1),
                     self.agent_id: np.array([i])}
                mean, log_std = sess.run(self.actor_output_target, feed_dict=feed_dict)
                mean = mean.squeeze(axis=0)
                log_std = log_std.squeeze(axis=0)
                std = np.exp(log_std)
                a = np.random.normal(loc=mean, scale=std)
            action.append(a)
        return action, agent_to_train, action_to_train, loglikelihood_to_train, state_to_train

    '''def collect_transitions(self, sess, max_episode_len=25):
        total_rewards = []
        states = np.zeros((self.horrizon+1, self.n_agent*self.state_dim))
        states_to_train = np.zeros((self.horrizon, self.state_dim))
        actions = np.zeros((self.horrizon, self.action_dim))
        rewards = np.zeros((self.horrizon, self.n_agent))
        are_non_terminal = np.zeros((self.horrizon, self.n_agent))
        actions_loglikelihood = np.zeros(self.horrizon)
        agent_to_train = np.zeros(self.horrizon, dtype=np.int32)
        episode_steps = 0
        for step in range(self.horrizon):
            if self.env_info['done']:
                state = self.env.reset()
                episode_steps = 0
                state = np.concatenate([j.reshape(1,-1) for j in state], axis=1).flatten()
                self.env_info['last_state'] = state
                self.env_info['done'] = False
                self.env_info['total_reward'] = 0
            state = self.env_info['last_state']
            states[step] = state
            
            action, agent, action_to_train, loglikelihood_to_train, s = self.sample_training_action(sess, state)
            agent_to_train[step] = agent
            states_to_train[step] = s
            state, reward, done, _ = self.env.step(action)
            episode_steps += 1
            state = np.concatenate([j.reshape(1,-1) for j in state], axis=1).flatten()
            self.env_info['last_state'] = state
            self.env_info['total_reward'] += reward[0]
            rewards[step]=reward
            actions[step,:] = action_to_train
            actions_loglikelihood[step] = loglikelihood_to_train
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
        agent_values_list = np.zeros((self.horrizon+1, self.n_agent))
        for agent_id in range(self.n_agent):
            agent_values = sess.run(self.values, feed_dict={self.states: states[:, agent_id*self.state_dim:(agent_id+1)*self.state_dim],
                                    self.agent_id: np.ones(self.horrizon+1)*agent_id}).squeeze(axis=1)
            agent_values_list[:, agent_id] = agent_values
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
        returns = returns[range(self.horrizon), agent_to_train]
        advantages = advantages[range(self.horrizon), agent_to_train]
        agent_id = agent_to_train
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-10)

        
        return states_to_train, actions, actions_loglikelihood, returns, advantages, agent_id, np.mean(total_rewards)'''

    def collect_transitions(self, sess, max_episode_len=25):
        total_rewards = []
        states = np.zeros((self.horrizon+1, self.n_agent*self.state_dim))
        actions = np.zeros((self.horrizon, self.n_agent*self.action_dim))
        rewards = np.zeros((self.horrizon, self.n_agent))
        are_non_terminal = np.zeros((self.horrizon, self.n_agent))
        actions_loglikelihood = np.zeros((self.horrizon, self.n_agent))
        episode_steps = 0
        for step in range(self.horrizon):
            if self.env_info['done']:
                state = self.env.reset()
                episode_steps = 0
                state = np.concatenate([j.reshape(1,-1) for j in state], axis=1).flatten()
                self.env_info['last_state'] = state
                self.env_info['done'] = False
                self.env_info['total_reward'] = 0
            state = self.env_info['last_state']
            states[step] = state
            
            action = []
            loglikelihood = []
            for i in range(self.n_agent):
                obs_start = i*self.state_dim
                obs_end = (i+1)*self.state_dim
                a, ll = self.sample_action(sess, state[obs_start: obs_end], i)
                action.append(a)
                loglikelihood.append(ll)
            state, reward, done, _ = self.env.step(action)
            episode_steps += 1
            state = np.concatenate([j.reshape(1,-1) for j in state], axis=1).flatten()
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
        for agent_id in range(self.n_agent):
            agent_values = sess.run(self.values, feed_dict={self.states: states[:, agent_id*self.state_dim:(agent_id+1)*self.state_dim],
                                    self.agent_id: np.ones(self.horrizon+1)*agent_id}).squeeze(axis=1)
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
            #advantages[:, agent_id] = (advantages[:, agent_id] - advantages[:, agent_id].mean()) / (advantages[:, agent_id].std() + 1e-9)

        states = np.concatenate([states[:-1, agent_id*self.state_dim:(agent_id+1)*self.state_dim] for agent_id in range(self.n_agent)])
        actions = np.concatenate([actions[:, agent_id*self.action_dim:(agent_id+1)*self.action_dim] for agent_id in range(self.n_agent)])
        actions_loglikelihood = np.concatenate([actions_loglikelihood[:, agent] for agent in range(self.n_agent)])
        returns = np.concatenate([returns[:, agent] for agent in range(self.n_agent)])
        advantages = np.concatenate([advantages[:, agent] for agent in range(self.n_agent)])
        agent_id = np.concatenate([agent*np.ones(self.horrizon) for agent in range(self.n_agent)])
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-10)

        
        return states, actions, actions_loglikelihood, returns, advantages, agent_id, np.mean(total_rewards)
    
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
                obs_start = i*self.state_dim
                obs_end = (i+1)*self.state_dim
                a, _ = self.sample_action(sess, state[obs_start:obs_end], i)
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
            return states, actions, rewards, infos