import os
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp-id', default=0, type=int)
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_arguments()
	algorithms = ["A2C", "QRA2C"]
	envs = ["CartPole-v0", "LunarLander-v2"]
	n_episodes = {"CartPole-v0": 2000, "LunarLander-v2": 20000}
	for algo in algorithms:
		for env in envs:
			print("************************")
			print("Running {} on {}.".format(algo, env))
			print("************************")
			os.system("python main.py --env {} --model {} --N 1 --actor-lr 1e-4 --critic-lr 1e-4 --train-episodes {} --n-quantile 200 --kappa 1e-6 --step 10 --log-dir exp-{}-log --model-dir exp-{}-model --plot-dir exp-{}-plot --device {}".format(env, algo, n_episodes[env], args.exp_id, args.exp_id, args.exp_id, args.exp_id%2))
