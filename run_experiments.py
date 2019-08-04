import os
import argparse
from collections import defaultdict

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp-id', default=0, type=int)
	parser.add_argument('--lr', default=2.5e-4, type=float)
	return parser.parse_args()

train_episodes = defaultdict(lambda : 500)
train_episodes["Ant-v2"] = 1500
# exp-0~4 100 quantiles 0.01 kappa, 50-percentile
# exp-9 kappa 1e-3 n-quantile 200
# exp-10 0.99 gamma
if __name__ == '__main__':
	args = parse_arguments()
	algorithms = ["MQRPPO"]
	# "PartiallyObservableAnt",    "PartiallyObservableHalfCheetah"
	envs = ["simple_spread_modified"]
	#envs = [ "PartiallyObservableWalker2d","PartiallyObservableAnt", "PartiallyObservableHalfCheetah", "PartiallyObservableHopper" ]
	for env in envs:
		for algo in algorithms:
			print("************************")
			print("Running {} on {}.".format(algo, env))
			print("************************")
			cmd = "python mpe_main.py --env {}".format(env)
			cmd += " --model {}".format(algo)
			cmd += " --train-episodes {}".format(6000)
			cmd += " --log-dir exp-{}-log".format(args.exp_id)
			cmd += " --model-dir exp-{}-model".format(args.exp_id)
			cmd += " --plot-dir exp-{}-plot".format(args.exp_id)
			cmd += " --gamma 0.99"
			#cmd += " --restore"

			os.system(cmd)
