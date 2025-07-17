import time
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils_bandit import *

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Computer Modern'

fontsize = 25
legendsize = 20
ticksize = 17.5
linewidth = 2.5
markersize = 10
markeredgewidth = 4
axissize = 17.5

plt.rcParams['xtick.labelsize'] = ticksize
plt.rcParams['ytick.labelsize'] = ticksize


CMAP = plt.colormaps["coolwarm"]
COLORS = (
	('blue', 'lightsteelblue', 'darkblue'),
	('red', 'salmon', 'darkred'),
)

MARKERS = (('o','s'), ('<', 'x'))
densely_dotted = (0, (1, 1))
LINESTYLES = (('dashed', 'dotted'), ('solid', densely_dotted))

PLOT_DIR = 'plots'

# Plot regret vs horizon T
def plot_regret_vs_horizon(expt):
	REGRET_MAX = 25

	print('Expt: %s' % expt)

	algo2 = 'greedy'
	if expt == 'ucb':
		algo1 = 'ucb|0'
	elif expt == 'egreedy':
		algo1 = 'egreedy|0|1'

	settings = [algo1, algo2]
	A = len(settings)

	horizons = np.array([25, 50, 100, 200, 300, 400, 500])
	H = len(horizons)
	repeat = int(1e4)

	us = np.array([0.2, 0.8])
	K = len(us)
	samplers = tuple(SamplerBernoulli(u) for u in us)

	rewards_separate = np.zeros((A, H, repeat)) # cumulative reward
	rewards_joint = np.zeros((A, H, repeat))

	tic = time.time()
	for h in range(H):
		T = horizons[h]

		print('T: %d/%d' % (h+1, H))

		for r in range(repeat):
			if r % 10000 == 0:
				print('  [%d/%d] %.1f sec' % (r+1, repeat, time.time() - tic))

			(reward_separate1, _) = simulator_joint(samplers, T, [algo1])
			(reward_separate2, _) = simulator_joint(samplers, T, [algo2])
			(reward_joint, _) = simulator_joint(samplers, T, settings)

			rewards_separate[:, h, r] = np.sum(reward_separate1), np.sum(reward_separate2)
			rewards_joint[:, h, r] = np.sum(reward_joint, axis=1)

	(fig, ax) = plt.subplots()
	for a in range(A):
		if settings[a].startswith('ucb'):
			settings[a] = 'UCB'
		elif settings[a].startswith('egreedy'):
			settings[a] = r'$\epsilon$-greedy'

	for a in range(A):
		ax.errorbar(horizons, np.max(us) * horizons - np.mean(rewards_separate[a, :, :], axis=1), 
					yerr=np.std(rewards_separate[a, :, :], axis=1) / np.sqrt(repeat),
					label='%s' % settings[a], linestyle=LINESTYLES[a][1], color=COLORS[a][1], marker=MARKERS[a][1],
					linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
	for a in range(A):
		ax.errorbar(horizons, np.max(us) * horizons - np.mean(rewards_joint[a, :, :], axis=1), 
					 yerr=np.std(rewards_joint[a, :, :], axis=1) / np.sqrt(repeat),
					 label= '%s (data sharing)' % settings[a], linestyle=LINESTYLES[a][0], color=COLORS[a][0], marker=MARKERS[a][0],
					 linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)

	ax.set_xlabel('Horizon (' + r'$T$' + ')', fontsize=axissize)
	ax.set_ylabel('Regret', fontsize=axissize)
	ax.set_ylim([0, REGRET_MAX])
	ax.legend(fontsize=legendsize)
	fig.set_size_inches(10, 5) # [W, H]
	plt.tight_layout()
	plt.savefig('%s/%s_greedy_vs_horizon_repeat%d.pdf' % (PLOT_DIR, expt, repeat))
	# plt.show()

# Plot regret difference and probability of correct comparison
def plot_2d(expt):
	print('Expt: %s' % expt)

	T = 100
	repeat =int(1e4)

	us = np.array([0.2, 0.8])
	K = len(us)
	samplers = [SamplerBernoulli(u) for u in us]

	# vary \alpha
	if expt == 'ucb':
		setting_str = 'ucb|%.1f'

		param_str = r'\alpha'
		params = np.linspace(0, 1, 11)
		xlabel_str = r'Exploration level ($%s$)' % param_str
	elif expt == 'egreedy':
		setting_str = 'egreedy|%.1f|1'

		param_str = r'\alpha'
		params = np.linspace(0, 1, 11) #[1:]# exclude uniform # t^\alpha
		xlabel_str = r'Exploration level ($%s$)' % param_str
	elif expt == 'ts':
		setting_str = 'ts|%d|%d|%d|%d'

		param_str = r'\gamma'
		params = np.linspace(0, 1, 6)
		xlabel_str = r'Prior misspecification ($%s$)' % param_str
		prior_size = 5

	print('Parameters:')
	print(params)
	S = len(params) # number of settings
	params_labels = convert_to_label(params)

	## run jointly
	# only fill upper triangle + diagonal
	rewards_mtx = np.empty((S, S), dtype=np.ndarray)
	rewards_mtx[np.tril_indices(S, -1)] = np.nan

	tic = time.time()
	for a in range(S):
		for b in range(a, S):
			print('(%d, %d) %.1f sec' % (a+1, b+1, time.time() - tic))
			rewards_mtx[a, b] = np.zeros((2, repeat)) # one row for algo A and one row for algo B

			for r in range(repeat):
				if r % 10000 == 0: print(' %d/%d %.1f sec' % (r+1, repeat, time.time() - tic))
				if expt == 'ts':
					(count_a, count_b) = np.round([prior_size * params[a], prior_size * params[b]])
					algos_pair = [setting_str % (count_a, prior_size - count_a, prior_size - count_a, count_a),
								  setting_str % (count_b, prior_size - count_b, prior_size - count_b, count_b)]
				else:
					algos_pair = [setting_str % params[a], setting_str % params[b]]
				(rewards, _) = simulator_joint(samplers, T, algos_pair)
				rewards_mtx[a, b][:, r] = np.sum(rewards, axis=1)

	## run individually
	rewards_single = np.zeros((S, repeat))
	for a in range(S):
		param = params[a]
		for r in range(repeat):
			if expt == 'ts':
				count = np.round(prior_size * param)
				algo = setting_str % (count, prior_size - count, prior_size - count, count)
			else:
				algo = setting_str % param
			rewards_single[a, r] = np.sum(simulator_joint(samplers, T, [algo])[0])

	baselines_single = np.zeros((S, 2)) # (mean, str)

	for a in range(S):
		baselines_single[a, :] = np.mean(rewards_single[a, :]), np.std(rewards_single[a, :]) / np.sqrt(repeat)

	# plot bias (2D)
	rewards_mtx_mean = np.full((S, S), np.nan)
	# upper triangle
	for a in range(S):
		for b in range(a+1, S):
			rewards_mtx_mean[a, b] = np.mean(rewards_mtx[a, b][0, :]) # a
	# diagonal
	for a in range(S):
		rewards_mtx_mean[a, a] = np.mean(rewards_mtx[a, a])
	# lower triangle
	for a in range(S):
		for b in range(0, a):
			rewards_mtx_mean[a, b] = np.mean(rewards_mtx[b, a][1, :]) # a

	## comparison probability (run individually)
	(fig, ax) = plt.subplots()
	rewards_mtx_comp = np.full((S, S), np.nan)
	for a in range(S):
		for b in range(S):
			rewards_mtx_comp[a, b] = np.mean(rewards_single[a, :] > rewards_single[b, :]) \
									 + 0.5 * np.mean(rewards_single[a, :] == rewards_single[b, :])

	im = ax.imshow(1-rewards_mtx_comp, cmap=CMAP, interpolation='none', origin='lower', vmin=0, vmax=1)
	fig.colorbar(im, ax=ax)

	ax.set_xlabel(r'Algo 2 ($%s_2$)' % param_str, fontsize=axissize) # B
	ax.set_ylabel(r'Algo 1 ($%s_1$)' % param_str, fontsize=axissize) # A
	ax.set_xticks(np.arange(S))
	ax.set_xticklabels(params_labels, rotation=45)
	ax.set_yticks(np.arange(S))
	ax.set_yticklabels(params_labels)

	plt.tight_layout()
	plt.savefig('%s/2d_percent_isolation_%s_T%d_repeat%d.pdf' % (PLOT_DIR, expt, T, repeat))

	## bias (joint)
	(fig, ax) = plt.subplots()
	diff = rewards_mtx_mean - np.transpose(rewards_mtx_mean)
	diff_max = np.max(np.abs(diff))
	im = ax.imshow(-diff, cmap=CMAP, interpolation='none', origin='lower', vmin=-diff_max, vmax=diff_max) # minus for regret
	fig.colorbar(im, ax=ax)
	ax.set_ylabel(r'Algo 1 ($%s_1$)' % param_str, fontsize=axissize) # A
	ax.set_xlabel(r'Algo 2 ($%s_2$)' % param_str, fontsize=axissize) # B

	ax.set_xticks(np.arange(S))
	ax.set_xticklabels(params_labels, rotation=45)
	ax.set_yticks(np.arange(S))
	ax.set_yticklabels(params_labels)
	plt.tight_layout()
	plt.savefig('%s/2d_bias_joint_%s_T%d_repeat%d.pdf' % (PLOT_DIR, expt, T, repeat))

	## comparison probability (joint)
	(fig, ax) = plt.subplots()
	rewards_mtx_comp = np.full((S, S), np.nan)
	for a in range(S): # upper triangle + diagonal
		for b in range(a, S):
			rewards_mtx_comp[a, b] = np.mean(rewards_mtx[a, b][0, :] > rewards_mtx[a, b][1, :]) \
									 + 0.5 * np.mean(rewards_mtx[a, b][0, :] == rewards_mtx[a, b][1, :]) # reward(a > b)
	for a in range(S): # lower triangle
		for b in range(0, a):
			rewards_mtx_comp[a, b] = np.mean(rewards_mtx[b, a][1, :] > rewards_mtx[b, a][0, :]) \
									 + 0.5 * np.mean(rewards_mtx[b, a][1, :] == rewards_mtx[b, a][0, :]) # a > b

	im = ax.imshow(1-rewards_mtx_comp, cmap=CMAP, interpolation='none', origin='lower', vmin=0, vmax=1)
	fig.colorbar(im, ax=ax)
	ax.set_ylabel(r'Algo 1 ($%s_1$)' % param_str, fontsize=axissize) # A
	ax.set_xlabel(r'Algo 2 ($%s_2$)' % param_str, fontsize=axissize) # B

	ax.set_xticks(np.arange(S))
	ax.set_xticklabels(params_labels, rotation=45)
	ax.set_yticks(np.arange(S))
	ax.set_yticklabels(params_labels)

	plt.tight_layout()
	plt.savefig('%s/2d_percent_joint_%s_T%d_repeat%d.pdf' % (PLOT_DIR, expt, T, repeat))

	# bias (isolation)
	(fig, ax) = plt.subplots()
	ax.errorbar(np.arange(S), T * np.max(us) - baselines_single[:, 0], yerr=baselines_single[:, 1], marker='o',
		linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
	ax.set_ylim([0, None])
	ax.set_xticks(np.arange(S))
	ax.set_xticklabels(params_labels, rotation=45)
	ax.set_xlabel(xlabel_str, fontsize=axissize)
	ax.set_ylabel('Regret', fontsize=axissize)
	ax.set_box_aspect(1)

	plt.tight_layout()
	plt.savefig('%s/2d_bias_isolation_%s_T%d_repeat%d.pdf' % (PLOT_DIR, expt, T, repeat))
	# plt.show()

def convert_to_label(params):
	def convert_to_label_single(param):
		if param == int(param):
			return '%d' % param
		return '%.1f' % param

	return [convert_to_label_single(param) for param in params]

if __name__ == '__main__':
	np.random.seed(0)

	if not os.path.isdir(PLOT_DIR):
			print('mkdir %s...' % PLOT_DIR)
			os.mkdir(PLOT_DIR)

	# Fig 3
	for expt in  ['ucb','egreedy']:
		plot_regret_vs_horizon(expt)

	# Figs 4, 5, 6
	for expt in ['ucb', 'egreedy', 'ts']:
		plot_2d(expt)
