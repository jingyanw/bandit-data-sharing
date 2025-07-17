# Choosing the Better Bandit Algorithm under Data Sharing: When Do A/B Experiments Work?

This is the code for the paper "Choosing the Better Bandit Algorithm under Data Sharing: When Do A/B Experiments Work?" ([link](https://arxiv.org/abs/2507.11891)).


## Code for Simulation
To replicate the simulation results reported in the paper, run the command:
```
python simulation.py
```

- The function `plot_regret_vs_horizon` runs the simulation for running two algorithms jointly (greedy paired with either epsilon-greedy or UCB) in Figure 3 of the paper.

- The function `plot_2d` runs the simulation for running a pair of algorithms jointly, and plot the expected bias and the probability of correct comparison, for epsilon-greedy, UCB, and Thompson sampling in Figure 4, Figure 5, and Figure 6 of the paper.


## Citation
```
@article{li2025sharing,
	author    = {Shuangning Li and Chonghuan Wang and Jingyan Wang},
	title     = {Choosing the Better Bandit Algorithm under Data Sharing: When Do A/B Experiments Work?},
	journal   = {arXiv preprint arXiv:2507.11891},
	year      = {2025},
}
```

## Contact
If you have any questions or feedback about the code or the paper, please contact Jingyan Wang (jingyanw@ttic.edu).
