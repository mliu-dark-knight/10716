import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from train import get_trainers, make_env
from tqdm import tqdm

class ImportMaddpgAgents():
    def __init__(self, model_path, env):
        self.graph = tf.Graph()
        self.config = tf.ConfigProto()
        self.config.intra_op_parallelism_threads = 1
        self.config.inter_op_parallelism_threads= 1
        self.sess = tf.Session(graph=self.graph, config=self.config)
        with self.graph.as_default():
            obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
            num_adversaries = min(env.n, arglist.num_adversaries)
            self.trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
            saver = tf.train.Saver()
            saver.restore(self.sess, model_path)
    def act(self, obs_n):
        with self.graph.as_default():
            with self.sess.as_default() as sess:
                action_n = [agent.action(obs) for agent, obs in zip(self.trainers,obs_n)]
                return action_n
    def close_sess(self):
        self.sess.close()

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--exp-id", type=int, default=0)
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="reproduce", help="name of the experiment")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="benchmark_files", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="learning_curves", help="directory where plot data is saved")
    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()
    env =  make_env("simple_spread", arglist, True)

    maddpg_team0_dir = os.path.join("exp-0", "simple_spread", "policy")
    maddpg_team_0 = ImportMaddpgAgents(maddpg_team0_dir, env)

    maddpg_team1_dir = os.path.join("exp-1", "simple_spread", "policy")
    maddpg_team_1 = ImportMaddpgAgents(maddpg_team1_dir, env)

    maddpg_team2_dir = os.path.join("exp-2", "simple_spread", "policy")
    maddpg_team_2 = ImportMaddpgAgents(maddpg_team0_dir, env)
    
    infos = []
    n_epi = 4000
    for epi in tqdm(range(n_epi), ncols=100):
        for step in range(25):
            obs_n = env.reset()
            action_0 = maddpg_team_0.act(obs_n)
            action_1 = maddpg_team_1.act(obs_n)
            action_2 = maddpg_team_2.act(obs_n)
            action = [action_1[0], action_1[1], action_1[2]]
            obs_n, rew_n, done_n, info_n = env.step(action)
            infos.append(info_n['n'])
            if all(done_n):
                break
    occ = []
    for info in infos:
        for info_ in info:
            occ.append(info_[3])
    print(sum(occ)/(len(occ)))
    maddpg_team_0.close_sess()
    maddpg_team_1.close_sess()
    maddpg_team_2.close_sess()
