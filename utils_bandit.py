import numpy as np

EPS = 1e-6
# Inputs:
# 	SETTINGS:
# 		greedy
# 		egreedy | alpha | C
# 		ucb | alpha
# 		ts|H1|T1|H2|T2|...|H_K|T_K (Thompson sampling with number of heads (1) and tails (0), a prior for each arm)
#
# Return:
# 	REWARDS: A x T matrix
# 	PULLS: A x T matrix
def simulator_joint(samplers, T, settings, eps=EPS):
	A = len(settings) # number of algorithms running together (2 for A/B testing)
	K = len(samplers)

	# parse TS
	# each prior is K x 2
	priors = [None] * A
	for a in range(A):
		setting = settings[a]
		if setting.startswith('ts'):
			config = setting.split('|')
			config = tuple(map(int, config[1:]))
			assert(len(config) == 2 * K)
			priors[a] = np.reshape(config, (K, 2))

	# shared data among algorithms
	sums = np.zeros(K) # for each arm
	counts = np.zeros(K) # for each arm

	rewards = np.zeros((A, T)) # for each algo
	pulls = np.zeros((A, T), dtype=int)

	# init: each algo pulls each arm once
	for a in range(A):
		setting = settings[a]

		for k in range(K):
			idx = k
			reward = samplers[idx].sample()

			sums[idx] += reward
			counts[idx] += 1

			rewards[a, k] = reward
			pulls[a, k] = idx

	for t in range(K, T):
		counts_round = np.zeros(K, dtype=int) # data collected at this timestep over all algos
		sums_round = np.zeros(K)

		history_size = np.sum(counts)

		for a in range(A):
			setting = settings[a]

			if setting.startswith('egreedy'):
				config = setting.split('|')
				alpha = float(config[1])
				C = float(config[2])

				prob_explore = C / np.power(history_size, 1-alpha)

				if np.random.uniform() < prob_explore: # random exploration
					idx = np.random.randint(K)

				else:
					idx = select_arm_greedy(counts, sums)

			elif setting == 'greedy':
				idx = select_arm_greedy(counts, sums)

			elif setting.startswith('ucb'):
				config = setting.split('|')
				alpha = float(config[1])
				idx = select_arm_ucb(counts, sums, alpha)
			elif setting.startswith('ts'):
				idx = select_arm_ts(counts, sums, priors[a])
			else:
				raise Exception('algo %s not implemented' % setting)

			reward = samplers[idx].sample()

			rewards[a, t] = reward
			pulls[a, t] = idx

			sums_round[idx] += reward
			counts_round[idx] += 1

		# after all algorithms finish the current timestep
		sums = sums + sums_round
		counts = counts + counts_round

	return rewards, pulls

################################################
### A single bandit algorithm ###
################################################
# select the next arm given SUMS and COUNTS
def select_arm_ucb(counts, sums, alpha, const=2, eps=EPS):
	K = len(counts)
	history_size = np.sum(counts)

	if alpha == 0:
		ucbs = np.sqrt(const * np.log(history_size) / counts)
	else: # 0 < alpha <= 1
		ucbs = np.sqrt(const * (np.power(history_size, alpha) - 1) / counts / alpha)

	means = sums / counts
	means_ucb = means + ucbs

	means_ucb_max = np.max(means_ucb)
	idxs = np.where(means_ucb > means_ucb_max - eps)[0]
	idx = np.random.choice(idxs) # break ties randomly
	return idx

# Select an arm based on the history
def select_arm_greedy(counts, sums, eps=EPS):
	means = sums / counts

	means_max = np.max(means)
	idxs = np.where(means > means_max - eps)[0]
	idx = np.random.choice(idxs) # break ties randomly
	return idx

# Thompson sampling
# Rewards must be {0, 1} only
# Inputs:
# 	PRIOR: K x 2
def select_arm_ts(counts, sums, priors, eps=EPS):
	# wins, losses
	samples = np.random.beta(priors[:, 0] + sums + 1, priors[:, 1] + counts - sums + 1)
	sample_max = np.max(samples)

	idxs = np.where(samples > sample_max - eps)[0]
	idx = np.random.choice(idxs) # break ties randomly
	return idx

################################################
### Distributions ###
################################################
class SamplerBernoulli:
	def __init__(self, mean):
		self.mean = mean

	def sample(self, size=1):
		if size == 1:
			return np.random.uniform() < self.mean
		else:
			return np.random.uniform(size=size) < self.mean

class SamplerNormal:
	def __init__(self, mean=0, std=1):
		self.mean = mean
		self.std = std

	def sample(self, size=1):
		if size == 1:
			return np.random.normal(self.mean, self.std)
		else:
			return np.random.normal(self.mean, self.std, size=size)
