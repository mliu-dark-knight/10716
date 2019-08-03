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
# exp-0 100 quantile 1 kappa
# exp-1 200 quantile 1 kappa
# exp-2 100 quantile 0.01 kappa
# exp-3 200 quantile 0.01 kappa
# exp-4 200 quantile 0.01 kappa hidden-dims 256
# exp-5 100 quantile 0.01 kappa no l2 reg
# exp-6 MSQRPPO 0.01 kappa no l2 reg
# exp-7 100 quantile 0.01 kappa no l2 reg 0.1 quantile
## exp-8 100 quantile 0.01 kappa no l2 reg 0.01 quantile
# exp-9 100 quantile 0.5 kappa no l2 reg
# exp-10 100 quantile 0.5 kappa no l2 reg
if __name__ == '__main__':
	args = parse_arguments()
	algorithms = ["MQRPPO"]
	# "PartiallyObservableAnt",    "PartiallyObservableHalfCheetah"
	envs = [ "simple_spread" ]
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
			cmd += " --n-quantile 100 --step 1 --kappa 0.01 --quantile 0.5"

			os.system(cmd)
