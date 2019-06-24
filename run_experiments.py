import os
import argparse
from collections import defaultdict

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp-id', default=0, type=int)
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_arguments()
	algorithms = ["SQRPPO"]
	#algorithms = ["PPO", "QRPPO"]
	#envs = ["HalfCheetah-v2"]
	#envs = ["Hopper-v2", "InvertedPendulum-v2"]
	train_episodes = defaultdict(lambda x: 1000)
	train_episodes["HalfCheetah-v2"] = 3000
	envs = ["InvertedPendulum-v2", "HalfCheetah-v2","Hopper-v2",  "InvertedDoublePendulum-v2"]
	for env in envs:
		for algo in algorithms:
			print("************************")
			print("Running {} on {}.".format(algo, env))
			print("************************")
			cmd = "python main.py --env {}".format(env)
			cmd += " --model {}".format(algo)
			cmd += " --train-episodes {}".format(1000)
			cmd += " --log-dir exp-{}-log".format(args.exp_id)
			cmd += " --model-dir exp-{}-model".format(args.exp_id)
			cmd += " --plot-dir exp-{}-plot".format(args.exp_id)
			cmd += " --device -1"
			os.system(cmd)
