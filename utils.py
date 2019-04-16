import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NUM_POINTS = 300.0

def append_summary(progress_fd, summary):
	progress_fd.write(summary + '\n')
	progress_fd.flush()


def plot(prefix, rewards):
	x_gap = len(rewards) / NUM_POINTS
	x_vals = np.arange(0, len(rewards), x_gap).astype(int)
	rewards = np.array(rewards)

	name, axis_label, func = 'avg', 'Reward Average (next 100)', points_avg
	y_vals = func(rewards, x_vals)
	plt.plot(x_vals + 1, y_vals)
	plt.xlabel('Unit of training (Actions in W1, Episodes in W2)')
	plt.ylabel(axis_label)
	plt.grid(which='Both')
	plt.tight_layout()
	plt.savefig(prefix + '_' + name + '_' + 'lin.png')
	plt.close()


def points_avg(rewards, x_vals):
	return np.array([np.sum(rewards[val:min(len(rewards), val + 100)]) \
					 / float(min(len(rewards) - val, 100)) for val in x_vals])
