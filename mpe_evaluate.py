import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from m3ddpg import M3DDPGAgentTrainer
import tensorflow.contrib.layers as layers
from tqdm import tqdm
from algorithms.MSQRPPO import MSQRPPO
from algorithms.SMPPO import SMPPO
from tensorflow.python import pywrap_tensorflow

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_maddpg_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.maddpg_bad_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.maddpg_good_policy=='ddpg')))
    return trainers

def get_m3ddpg_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = M3DDPGAgentTrainer
    for i in range(num_adversaries):
        policy_name = arglist.m3ddpg_bad_policy
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    for i in range(num_adversaries, env.n):
        policy_name = arglist.m3ddpg_good_policy
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    return trainers


class ImportMaddpgAgents():
    def __init__(self, model_path, env):
        self.graph = tf.Graph()
        self.config = tf.ConfigProto()
        self.config.intra_op_parallelism_threads = 1
        self.config.inter_op_parallelism_threads= 1
        self.sess = tf.Session(graph=self.graph, config=self.config)
        with self.graph.as_default():
            with self.sess.as_default() as sess:
                U.initialize()
                obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
                num_adversaries = min(env.n, arglist.num_adversaries)
                self.trainers = get_maddpg_trainers(env, num_adversaries, obs_shape_n, arglist)
                saver = tf.train.Saver()

                saver.restore(self.sess, model_path)
    def act(self, obs_n):
        with self.graph.as_default():
            with self.sess.as_default() as sess:
                action_n = [agent.action(obs) for agent, obs in zip(self.trainers,obs_n)]
                return action_n
    def close_sess(self):
        self.sess.close()

class ImportM3ddpgAgents():
    def __init__(self, model_path, env):
        self.graph = tf.Graph()
        self.config = tf.ConfigProto()
        self.config.intra_op_parallelism_threads = 1
        self.config.inter_op_parallelism_threads= 1
        self.sess = tf.Session(graph=self.graph, config=self.config)
        with self.graph.as_default():
            with self.sess.as_default() as sess:
                U.initialize()
                obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
                num_adversaries = min(env.n, arglist.num_adversaries)
                self.trainers = get_m3ddpg_trainers(env, num_adversaries, obs_shape_n, arglist)
                saver = tf.train.Saver()
                saver.restore(self.sess, model_path)
    def act(self, obs_n):
        with self.graph.as_default():
            with self.sess.as_default() as sess:
                action_n = [agent.action(obs) for agent, obs in zip(self.trainers,obs_n)]
                return action_n
    def close_sess(self):
        self.sess.close()

class ImportMQRPPOAgent():
    def __init__(self, model_path, env):
        self.graph = tf.Graph()
        self.config = tf.ConfigProto()
        self.config.intra_op_parallelism_threads = 1
        self.config.inter_op_parallelism_threads= 1
        self.sess = tf.Session(graph=self.graph, config=self.config)
        with self.graph.as_default():
            with self.sess.as_default() as sess:
                self.agent = SMPPO(env, hidden_dims=[64, 64],
                                        kappa=1e-2,
                                        gamma=0.95,
                                        quantile=0.5)
                saver = tf.train.Saver()
                saver.restore(self.sess, os.path.join(model_path, 'model.ckpt'))
                #self.agent.load_state_filter(os.path.join(model_path, 'filter.npy'))
    def act(self, obs_n):
        action = []
        with self.sess.as_default() as sess:    
            with self.graph.as_default():
                state = np.concatenate([j.reshape(1,-1) for j in obs_n], axis=1).flatten()
                for agent in range(self.agent.n_agent):
                    agent_state = state[self.agent.state_dim*agent:self.agent.state_dim*(1+agent)]
                    action.append(self.agent.get_deterministic_action(sess, agent_state, agent))
        return action
    def close_sess(self):
        self.sess.close()

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--maddpg-good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--maddpg-bad-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--m3ddpg-good-policy", type=str, default="mmmaddpg", help="policy for good agents")
    parser.add_argument("--m3ddpg-bad-policy", type=str, default="mmmaddpg", help="policy of adversaries")
    parser.add_argument("--exp-id", type=int, default=0)
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    parser.add_argument("--adv-eps-s", type=float, default=1e-5, help="small adversarial training rate")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="reproduce", help="name of the experiment")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-name", type=str, default="", help="name of which training state and model are loaded, leave blank to load seperately")
    parser.add_argument("--load-good", type=str, default="", help="which good policy to load")
    parser.add_argument("--load-bad", type=str, default="", help="which bad policy to load")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="benchmark_files", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="learning_curves", help="directory where plot data is saved")
    return parser.parse_args()

def evaluate_simple_spread(agent_team_list, env, fixed_teammates = False):
    n_epi = 400
    infos = []
    for epi in range(n_epi):
        obs_n = env.reset()
        for step in range(25):
            action = []
            for idx, agent_team in enumerate(agent_team_list):
                action.append(agent_team.act(obs_n)[idx])
            if fixed_teammates:
                action[1] = np.zeros_like(action[1])
                action[2] = np.zeros_like(action[2])
            obs_n, rew_n, done_n, info_n = env.step(action)
            infos.append(info_n['n'])
            if all(done_n):
                break
    occ = []
    for info in infos:
        for info_ in info:
            occ.append(info_[3])
    return sum(occ)/len(occ)

def evaluate_simple_push(agent_team_list, env, fixed_adversary = False):
    n_epi = 400
    infos = []
    for epi in range(n_epi):
        obs_n = env.reset()
        for step in range(25):
            action = []
            for idx, agent_team in enumerate(agent_team_list):
                action.append(agent_team.act(obs_n)[idx])
            if fixed_adversary:
                action[0] = np.zeros_like(action[0])
            obs_n, rew_n, done_n, info_n = env.step(action)
            infos.append(info_n['n'])
            if all(done_n):
                break
    occ = []
    for info in infos:
        for info_ in info:
            occ.append(info_[0])
    return sum(occ)/len(occ)

def simple_spread(arglist):
    env =  make_env("simple_spread_modified", arglist, True)
    maddpg_teams = []
    m3ddpg_teams = []
    for i in range(3):
        maddpg_dir = os.path.join("exp-{}-model".format(i), "simple_spread_modified", "policy")
        maddpg_team = ImportMaddpgAgents(maddpg_dir, env)
        maddpg_teams.append(maddpg_team)
        m3ddpg_dir = os.path.join("exp-{}-model".format(i), "m3ddog_simple_spread_modified", "policy")
        m3ddpg_team = ImportM3ddpgAgents(m3ddpg_dir, env)
        m3ddpg_teams.append(m3ddpg_team)
    # No new agent
    scores = []
    for team_id in range(3):
        scores.append(evaluate_simple_spread([maddpg_teams[team_id], maddpg_teams[team_id], maddpg_teams[team_id]], env))
    print("maddpg no new agent {:.4f}+-{:.4f}, n-trail {}".format(np.mean(scores), np.std(scores)/np.sqrt(len(scores)), len(scores)))
    scores = []
    for team_id in range(3):
        scores.append(evaluate_simple_spread([m3ddpg_teams[team_id], m3ddpg_teams[team_id], m3ddpg_teams[team_id]], env))
    print("m3ddpg no new agent {:.4f}+-{:.4f}, n-trail {}".format(np.mean(scores), np.std(scores)/np.sqrt(len(scores)), len(scores)))
    # One new agent
    scores = []
    for eq in range(3):
        for neq in range(3):
            if neq == eq:
                continue
            for neq_pos in range(3):
                agent_list = [eq for i in range(0, neq_pos)]
                agent_list.append(neq)
                agent_list += [eq for i in range(neq_pos+1, 3)]
                agent_list = [maddpg_teams[i] for i in agent_list]
                scores.append(evaluate_simple_spread(agent_list, env))
    print("maddpg one new agent {:.4f}+-{:.4f}, n-trail {}".format(np.mean(scores), np.std(scores)/np.sqrt(len(scores)), len(scores)))
    scores = []
    for eq in range(3):
        for neq in range(3):
            if neq == eq:
                continue
            for neq_pos in range(3):
                agent_list = [eq for i in range(0, neq_pos)]
                agent_list.append(neq)
                agent_list += [eq for i in range(neq_pos+1, 3)]
                agent_list = [m3ddpg_teams[i] for i in agent_list]
                scores.append(evaluate_simple_spread(agent_list, env))
    print("m3ddpg one new agent {:.4f}+-{:.4f}, n-trail {}".format(np.mean(scores), np.std(scores)/np.sqrt(len(scores)), len(scores)))
    # Two new agent
    scores = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i==j or i==k or j==k:
                    continue
                agent_list = [maddpg_teams[i], maddpg_teams[j], maddpg_teams[k]]
                scores.append(evaluate_simple_spread(agent_list, env))
    print("maddpg one new agent {:.4f}+-{:.4f}, n-trail {}".format(np.mean(scores), np.std(scores)/np.sqrt(len(scores)), len(scores)))
    scores = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i==j or i==k or j==k:
                    continue
                agent_list = [m3ddpg_teams[i], m3ddpg_teams[j], m3ddpg_teams[k]]
                scores.append(evaluate_simple_spread(agent_list, env))
    print("m3ddpg one new agent {:.4f}+-{:.4f}, n-trail {}".format(np.mean(scores), np.std(scores)/np.sqrt(len(scores)), len(scores)))
    for team in maddpg_teams+m3ddpg_teams:
        team.close_sess()
if __name__ == '__main__':
    arglist = parse_args()
    '''
    env =  make_env("simple_push_modified", arglist, True)
    m3ddpg_teams = []
    for i in range(6,9):
        m3ddpg_dir = os.path.join("exp-{}-model".format(i), "m3ddpg_simple_push_modified", "policy")
        m3ddpg_team = ImportM3ddpgAgents(m3ddpg_dir, env)
        m3ddpg_teams.append(m3ddpg_team)
    # No new agent
    scores = []
    for team_id in range(3):
        for team_id_ in range(3):
            if team_id == team_id_:
                continue
            scores.append(evaluate_simple_push([m3ddpg_teams[team_id], m3ddpg_teams[team_id_]], env))
    print("m3ddpg vs maddpg one new agent {:.4f}+-{:.4f}, n-trail {}".format(np.mean(scores), np.std(scores)/np.sqrt(len(scores)), len(scores)))
    # One new agent
    #print(evaluate_simple_spread([m3ddpg_teams[0], m3ddpg_teams[0], m3ddpg_teams[0]], env))
    #team2_dir = os.path.join("exp-1-model", "SMPPO_simple_spread_modified")
    #team_2 = ImportMQRPPOAgent(team2_dir, env)
    
    #team0_dir = os.path.join("exp-0-model", "simple_spread_modified", "policy")
    #team_0 = ImportMaddpgAgents(team0_dir, env)
    #team1_dir = os.path.join("exp-1-model", "simple_spread_modified", "policy")
    #team_1 = ImportMaddpgAgents(team1_dir, env)'''
    env =  make_env("simple_spread_modified", arglist, True)
    maddpg_teams = []
    m3ddpg_teams = []
    for i in range(3):
        maddpg_dir = os.path.join("exp-{}-model".format(i), "simple_spread_modified", "policy")
        maddpg_team = ImportMaddpgAgents(maddpg_dir, env)
        maddpg_teams.append(maddpg_team)
        m3ddpg_dir = os.path.join("exp-{}-model".format(i), "m3ddpg_simple_spread_modified", "policy")
        m3ddpg_team = ImportM3ddpgAgents(m3ddpg_dir, env)
        m3ddpg_teams.append(m3ddpg_team)
    # No new agent
    scores = []
    for team_id in range(3):
        scores.append(evaluate_simple_spread([maddpg_teams[team_id], maddpg_teams[team_id], maddpg_teams[team_id]], env, True))
    print("maddpg no new agent {:.4f}+-{:.4f}, n-trail {}".format(np.mean(scores), np.std(scores)/np.sqrt(len(scores)), len(scores)))
    scores = []
    for team_id in range(3):
        scores.append(evaluate_simple_spread([m3ddpg_teams[team_id], m3ddpg_teams[team_id], m3ddpg_teams[team_id]], env, True))
    print("m3ddpg no new agent {:.4f}+-{:.4f}, n-trail {}".format(np.mean(scores), np.std(scores)/np.sqrt(len(scores)), len(scores)))

    
