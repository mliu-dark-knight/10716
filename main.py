import argparse
import os

import gym
import numpy as np
import tensorflow as tf

from algorithms.DDPG import DDPG
from algorithms.QRDDPG import QRDDPG
from algorithms.D3PG import D3PG
from algorithms.QRDQN import QRDQN
from algorithms.QRA2C import QRA2C
from algorithms.A2C import A2C
from algorithms.PPO import PPO
from algorithms.QRPPO import QRPPO
from algorithms.common import Replay_Memory
from utils import plot, append_summary


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', default='HandReach-v0', type=str,
	                    help='[FetchReach-v1, FetchSlide-v1, FetchPush-v1, FetchPickAndPlace-v1,'
	                         'HandReach-v0, HandManipulateBlock-v0, HandManipulateEgg-v0, HandManipulatePen-v0]')
	parser.add_argument('--model', default='QRDDPG', type=str, help='[DDPG, D3PG, QRDDPG]')
	parser.add_argument('--eval', default=False, action='store_true',
	                    help='Set this to False when training and True when evaluating.')
	parser.add_argument('--restore', default=False, action='store_true', help='Restore training')
	parser.add_argument('--reward-type', default='sparse', help='[sparse, dense]')
	parser.add_argument('--hidden-dims', default=[256, 256], type=int, nargs='+', help='Hidden dimension of network')
	parser.add_argument('--gamma', default=0.98, type=float, help='Reward discount')
	parser.add_argument('--lambd', default=0.96, type=float, help='discount for gae')
	parser.add_argument('--tau', default=1e-2, type=float, help='Soft parameter update tau')
	parser.add_argument('--kappa', default=1e-6, type=float, help='Kappa used in quantile Huber loss')
	parser.add_argument('--n-quantile', default=200, type=int, help='Number of quantile to approximate distribution')
	parser.add_argument('--actor-lr', default=1e-4, type=float, help='Actor learning rate')
	parser.add_argument('--critic-lr', default=1e-4, type=float, help='Critic learning rate')
	parser.add_argument('--n-atom', default=51, type=int, help='Number of atoms used in D3PG')
	parser.add_argument('--batch-size', default=256, type=int)
	parser.add_argument('--step', default=3, type=int, help='Number of gradient descent steps per episode')
	parser.add_argument('--epsilon', default=0.2, type=float, help='Exploration noise, fixed in D4PG')
	parser.add_argument('--train-episodes', default=100, type=int, help='Number of episodes to train')
	parser.add_argument('--save-episodes', default=100, type=int, help='Number of episodes to save model')
	parser.add_argument('--memory-size', default=1000000, type=int, help='Size of replay memory')
	parser.add_argument('--apply-her', default=False, action='store_true', help='Use HER or not')
	parser.add_argument('--n-goals', default=10, type=int, help='Number of goals to sample for HER')
	parser.add_argument('--C', default=1, type=int, help='Number of episodes to copy critic network to target network')
	parser.add_argument('--N', default=1, type=int, help='N step returns.')
	parser.add_argument('--plot-dir', default='plot', type=str, )
	parser.add_argument('--model-dir', default='model', type=str)
	parser.add_argument('--log-dir', default='log', type=str)
	parser.add_argument('--progress-file', default='progress.csv', type=str)
	parser.add_argument('--device', default=1, type=int, help='GPU device number')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_arguments()
	if not os.path.exists(args.plot_dir):
		os.makedirs(args.plot_dir)
	if not os.path.exists(args.model_dir):
		os.makedirs(args.model_dir)
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)
	model_path = os.path.join(os.path.join(args.model_dir, args.model + '_' + args.env), 'model.ckpt')
	log_path = os.path.join(args.log_dir, args.model + '_' + args.env)
	progress_file = os.path.join(log_path, args.progress_file)

	# Setting the session to allow growth, so it doesn't allocate all GPU memory.
	if args.device >= 0:
		os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
		device = '/gpu:0'
	else:
		device = '/cpu:0'
	if args.env in ['FetchReach-v1', 'FetchSlide-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1',
	                         'HandReach-v0', 'HandManipulateBlock-v0', 'HandManipulateEgg-v0', 'HandManipulatePen-v0']:
		environment = gym.make(args.env, reward_type=args.reward_type)
	else:
		environment = gym.make(args.env)

	tf.reset_default_graph()
	with tf.device(device):
		if args.eval:
			replay_memory = None
		else:
			replay_memory = Replay_Memory(memory_size=args.memory_size)
		if args.model == 'DDPG':
			agent = DDPG(environment, args.hidden_dims, replay_memory=replay_memory, gamma=args.gamma,
			             actor_lr=args.actor_lr, critic_lr=args.critic_lr, tau=args.tau, N=args.N)
		elif args.model == 'QRDDPG':
			agent = QRDDPG(environment, args.hidden_dims, replay_memory=replay_memory, gamma=args.gamma,
			               actor_lr=args.actor_lr, critic_lr=args.critic_lr, tau=args.tau, N=args.N, kappa=args.kappa,
			               n_quantile=args.n_quantile)
		elif args.model == 'D3PG':
			# Need a better way for setting v_min and v_max
			agent = D3PG(environment, args.hidden_dims, replay_memory=replay_memory, gamma=args.gamma,
			               actor_lr=args.actor_lr, critic_lr=args.critic_lr, tau=args.tau, N=args.N, n_atom = args.n_atom,
						   v_min=-100, v_max=100)
		elif args.model == 'QRDQN':
			agent = QRDQN(environment, args.hidden_dims, replay_memory=replay_memory, gamma=args.gamma,
			lr=args.actor_lr, tau=args.tau, N=args.N, kappa=args.kappa,n_quantile=args.n_quantile)
		elif args.model == 'QRA2C':
			agent = QRA2C(environment, args.hidden_dims, gamma=args.gamma,
			               actor_lr=args.actor_lr, critic_lr=args.critic_lr, tau=args.tau, N=args.N, kappa=args.kappa,
			               n_quantile=args.n_quantile)
		elif args.model == 'A2C':
			agent = A2C(environment, args.hidden_dims, gamma=args.gamma,
			               actor_lr=args.actor_lr, critic_lr=args.critic_lr, tau=args.tau, N=args.N)
		elif args.model == 'PPO':
			agent = PPO(environment, args.hidden_dims, gamma=args.gamma, lambd=args.lambd,
			               actor_lr=args.actor_lr, critic_lr=args.critic_lr, tau=args.tau, N=args.N)
		elif args.model == 'QRPPO':
			agent = QRPPO(environment, args.hidden_dims, gamma=args.gamma, lambd=args.lambd,
			               actor_lr=args.actor_lr, critic_lr=args.critic_lr, tau=args.tau, N=args.N, kappa=args.kappa,
			               n_quantile=args.n_quantile)
		else:
			raise NotImplementedError

	gpu_ops = tf.GPUOptions(per_process_gpu_memory_fraction=0.25, allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops, allow_soft_placement=True)

	saver = tf.train.Saver()
	summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

	with tf.Session(config=config) as sess:
		if args.eval or args.restore:
			saver.restore(sess, model_path)
			if not args.eval:
				progress_fd = open(progress_file, 'r')
				start_episode = len(progress_fd.readlines()) - 1
				progress_fd.close()
				progress_fd = open(progress_file, 'a')
		else:
			progress_fd = open(progress_file, 'w')
			append_summary(progress_fd, 'episode, total-reward, actor-loss, critic-loss')
			progress_fd.flush()
			start_episode = 0
			tf.global_variables_initializer().run()
		if not args.eval:
			total_rewards = agent.train(
				sess, saver, summary_writer, progress_fd, model_path, batch_size=args.batch_size, step=args.step,
				train_episodes=args.train_episodes, start_episode=start_episode, save_episodes=args.save_episodes,
				epsilon=args.epsilon, apply_her=args.apply_her, n_goals=args.n_goals)
			progress_fd.close()
			plot(os.path.join(args.plot_dir, args.model + '_' + args.env), np.array(total_rewards) + 1e-10)
		else:
			agent.generate_episode(epsilon=0.0, render=True)
