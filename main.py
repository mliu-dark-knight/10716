import argparse
import os

import gym

from algorithms.DDPG import *
from utils import plot


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env', default='FetchReach-v1', type=str,
	                    help='[FetchReach-v1, FetchSlide-v1, FetchPush-v1, FetchPickAndPlace-v1]')
	parser.add_argument('--model', default='algorithms', type=str, help='[algorithms, D3PG, IQDDPG]')
	parser.add_argument('--eval', default=False, action='store_true',
	                    help='Set this to False when training and True when evaluating.')
	parser.add_argument('--restore', default=False, action='store_true', help='Restore training')
	parser.add_argument('--hidden-dims', default=[256, 256], type=list, help='Hidden dimension of network')
	parser.add_argument('--gamma', default=1.0, type=float, help='Reward discount')
	parser.add_argument('--tau', default=1e-2, type=float, help='algorithms soft parameter update tau')
	parser.add_argument('--actor-lr', default=1e-4, type=float, help='Actor learning rate')
	parser.add_argument('--critic-lr', default=1e-3, type=float, help='Critic learning rate')
	parser.add_argument('--batch-size', default=128, type=int)
	parser.add_argument('--step', default=10, type=int, help='Number of gradient descent steps per episode')
	parser.add_argument('--epsilon', default=0.3, type=float, help='Exploration noise, fixed in D4PG')
	parser.add_argument('--train-episodes', default=1000, type=int, help='Number of episodes to train')
	parser.add_argument('--save-episodes', default=100, type=int, help='Number of episodes to save model')
	parser.add_argument('--memory-size', default=40000, type=int, help='Size of replay memory')
	parser.add_argument('--C', default=1, type=int, help='Number of episodes to copy critic network to target network')
	parser.add_argument('--N', type=int, default=10, help='N step returns.')
	parser.add_argument('--her-reward', type=float, default=-0.1, help='HER reward')
	parser.add_argument('--delta', type=float, default=0.1,
	                    help='Radius around goal out of which the agent will receive reward of -0.1')
	parser.add_argument('--plot-dir', type=str, default='plot/')
	parser.add_argument('--plot-prefix', type=str, default='DDPG')
	parser.add_argument('--model-path', default='model/DDPG/model.ckpt')
	parser.add_argument('--device', default=3, help='GPU device number')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_arguments()
	if not os.path.exists(args.plot_dir):
		os.makedirs(args.plot_dir)

	# Setting the session to allow growth, so it doesn't allocate all GPU memory.
	if args.device >= 0:
		os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
		device = '/gpu:0'
	else:
		device = '/cpu:0'

	environment = gym.make(args.env)

	tf.reset_default_graph()
	with tf.device(device):
		if args.eval:
			replay_memory = None
		else:
			replay_memory = Replay_Memory(memory_size=args.memory_size)
		if args.model == 'algorithms':
			agent = DDPG(environment, args.hidden_dims, replay_memory=replay_memory, gamma=args.gamma,
			             actor_lr=args.actor_lr, critic_lr=args.critic_lr, tau=args.tau, N=args.N,
			             her_reward=args.her_reward, delta=args.delta)
		else:
			raise NotImplementedError

	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops, allow_soft_placement=True)

	saver = tf.train.Saver()
	with tf.Session(config=config) as sess:
		if args.eval or args.restore:
			saver.restore(sess, args.model_path)
		else:
			sess.run(tf.global_variables_initializer())
		if not args.eval:
			if args.model == 'algorithms':
				total_rewards = agent.train(
					sess, saver, args.model_path, batch_size=args.batch_size, step=args.step,
					train_episodes=args.train_episodes, save_episodes=args.save_episodes, epsilon=args.epsilon)
			else:
				raise NotImplementedError
			plot(os.path.join(args.plot_dir, args.plot_prefix), np.array(total_rewards) + 1e-10)
		else:
			if args.model == 'algorithms':
				agent.generate_episode(epsilon=0.0, render=True)
			else:
				raise NotImplementedError
