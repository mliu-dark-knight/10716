import os
import argparse
from collections import defaultdict

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp-id', default=0, type=int)
	parser.add_argument('--lr', default=2.5e-4, type=float)
	return parser.parse_args()
# 3000: 6M
train_episodes = defaultdict(lambda : 500)
train_episodes["Ant-v2"] = 1500


if __name__ == '__main__':
	args = parse_arguments()
	algorithms = ["QRPPO","PPO"]
	# "PartiallyObservableAnt",    "PartiallyObservableHalfCheetah"
	envs = [ "Walker2d-v2","Ant-v2","HalfCheetah-v2", "Hopper-v2" ]
	#envs = [ "PartiallyObservableWalker2d","PartiallyObservableAnt", "PartiallyObservableHalfCheetah", "PartiallyObservableHopper" ]
	for env in envs:
		for algo in algorithms:
			print("************************")
			print("Running {} on {}.".format(algo, env))
			print("************************")
			cmd = "python main.py --env {}".format(env)
			cmd += " --model {}".format(algo)
			cmd += " --train-episodes {}".format(train_episodes[env])
			cmd += " --log-dir exp-{}-log".format(args.exp_id)
			cmd += " --model-dir exp-{}-model".format(args.exp_id)
			cmd += " --plot-dir exp-{}-plot".format(args.exp_id)
			#cmd += " --actor-lr {} --critic-lr {}".format(args.lr, args.lr)

			os.system(cmd)
