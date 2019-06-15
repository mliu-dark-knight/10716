import os
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp-id', default=0, type=int)
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_arguments()
	algorithms = ["PPO", "QRPPO"]
	envs = ["CartPole-v0", "LunarLander-v2", "Acrobot-v1", "Pendulum-v0"]
	n_episodes = {"CartPole-v0": 2000,
				  "LunarLander-v2": 30000,
				  "Acrobot-v1":5000,
				  "Pendulum-v0":20000}
	for algo in algorithms:
		for env in envs:
			print("************************")
			print("Running {} on {}.".format(algo, env))
			print("************************")
			cmd = "python main.py --env {}".format(env)
			cmd += " --model {}".format(algo)
			cmd += " --train-episodes {}".format(n_episodes[env])
			cmd += " --log-dir exp-{}-log".format(args.exp_id)
			cmd += " --model-dir exp-{}-log".format(args.exp_id)
			cmd += " --plot-dir exp-{}-log".format(args.exp_id)
			os.system(cmd)
