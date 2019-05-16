import argparse
import os
import numpy as np
import tensorflow as tf
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from algorithms.QRDQN import QRDQN, MultiagentWrapper
from algorithms.common import Replay_Memory
from utils import plot, append_summary


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag', type=str,
                        help='[simple_tag]')
    parser.add_argument('--model', default='QRDQN', type=str, help='[QRDQN]')
    parser.add_argument('--eval', default=False, action='store_true',
                        help='Set this to False when training and True when evaluating.')
    parser.add_argument('--restore', default=False, action='store_true', help='Restore training')
    parser.add_argument('--reward-type', default='sparse', help='[sparse, dense]')
    parser.add_argument('--hidden-dims', default=[256, 256], type=int, nargs='+', help='Hidden dimension of network')
    parser.add_argument('--gamma', default=1.0, type=float, help='Reward discount')
    parser.add_argument('--tau', default=1e-2, type=float, help='Soft parameter update tau')
    parser.add_argument('--kappa', default=1.0, type=float, help='Kappa used in quantile Huber loss')
    parser.add_argument('--n-quantile', default=32, type=int, help='Number of quantile to approximate distribution')
    parser.add_argument('--actor-lr', default=1e-4, type=float, help='Actor learning rate')
    parser.add_argument('--critic-lr', default=1e-4, type=float, help='Critic learning rate')
    parser.add_argument('--n-atom', default=51, type=int, help='Number of atoms used in D3PG')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--step', default=100, type=int, help='Number of gradient descent steps per episode')
    parser.add_argument('--epsilon', default=0.2, type=float, help='Exploration noise, fixed in D4PG')
    parser.add_argument('--train-episodes', default=100, type=int, help='Number of episodes to train')
    parser.add_argument('--save-episodes', default=100, type=int, help='Number of episodes to save model')
    parser.add_argument('--memory-size', default=1000000, type=int, help='Size of replay memory')
    parser.add_argument('--apply-her', default=False, action='store_true', help='Use HER or not')
    parser.add_argument('--n-goals', default=10, type=int, help='Number of goals to sample for HER')
    parser.add_argument('--C', default=1, type=int, help='Number of episodes to copy critic network to target network')
    parser.add_argument('--N', default=10, type=int, help='N step returns.')
    parser.add_argument('--plot-dir', default='plot', type=str, )
    parser.add_argument('--model-dir', default='model', type=str)
    parser.add_argument('--log-dir', default='log', type=str)
    parser.add_argument('--progress-file', default='progress.csv', type=str)
    parser.add_argument('--device', default=1, type=int, help='GPU device number')
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
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
    scenario = scenarios.load(args.env+".py").Scenario()
    world = scenario.make_world()
    environment = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                                scenario.observation, scenario.benchmark_data)
    environment.discrete_action_input = True
    n_agent = len(environment.agents)
    tf.reset_default_graph()
    with tf.device(device):
        agent_params = []
        for a in range(n_agent):
            if args.eval:
                replay_memory = None
            else:
                replay_memory = Replay_Memory(memory_size=args.memory_size)
            agent_params.append(dict(hidden_dims=args.hidden_dims,
                                     replay_memory=replay_memory,
                                     gamma=args.gamma,
                                     lr=args.actor_lr,
                                     tau=args.tau,
                                     N=args.N,
                                     kappa=args.kappa,
                                     n_quantile=args.n_quantile,
                                     scope_pre="agent{}_".format(a)))
        agent_wrapper = MultiagentWrapper(environment, n_agent, agent_params)
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
            append_summary(progress_fd, 'episode,first-agent-reward')
            progress_fd.flush()
            start_episode = 0
            tf.global_variables_initializer().run()
        if not args.eval:
            total_rewards = agent_wrapper.train(
                sess, saver, summary_writer, progress_fd, model_path, batch_size=args.batch_size, step=args.step,
                train_episodes=args.train_episodes, start_episode=start_episode, save_episodes=args.save_episodes,
                epsilon=args.epsilon, max_episode_len=args.max_episode_len)
            progress_fd.close()
            plot(os.path.join(args.plot_dir, args.model + '_' + args.env), np.array(total_rewards) + 1e-10)
        else:
            agent.generate_episode(epsilon=0.0,
                                   max_episode_len=args.max_episode_len,
                                   render=True)
