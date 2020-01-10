# encoding: utf-8
import numpy as np
from matplotlib import pylab as plt


# Results for 1000 samples, varying number of chains
# Times in seconds
n_chains = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
chains_results = {
    'Numpy': [3, 3.1, 3.1, 3.5, 7.4, 47.6, 477],
    'JAX': [0.6, 1.4, 1.7, 3.1, 1.5, 4.3, 26.6],
    'JAX (already compiled)': [0.1, 0.1, 0.1, 0.1, 0.2, 3.3, 25.4],
    'TFP': [7.7, 7.7, 7.8, 8.5, 12.9, np.nan, np.nan],
    'TFP (with XLA compilation)': [3.8, 4.2, 4.4, 5.8, 8.3, 45.5, np.nan]
}

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
for key, values in chains_results.items():
    ax.plot(n_chains, values, label=key)
ax.set_xlabel("Number of chains", fontsize=22, fontname="Source Code Pro")
ax.set_ylabel("Time (s)", fontsize=22, fontname="Source Code Pro")
fig.suptitle("Sampling 1,000 samples from a 4 components Gaussian mixture", fontsize=18, fontname="Source Code Pro")

ax.set_xscale('log')
ax.set_yscale('log')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.legend(frameon=False)

plt.savefig('chains.png', bbox_inches='tight', transparent=True)


# Results for 1000 chains, varying number of samples
# Times in seconds
n_samples = [1, 10, 100, 1_000, 10_000, 100_000]
samples_results = {
    'Numpy': [0.5, 0.5, 0.7, 3.5, 31.6, np.nan],
    'JAX': [3.6, 3.6, 3.6, 3.7, 4.7, 14.1],
    'JAX (already compiled)': [0.001, 0.01, 0.01, 0.1, 1, 10.4],
    'TFP': [1.9, 2, 2.5, 8.3, 68 , np.nan],
    'TFP (with XLA compilation)': [5.9, 5.8, 5.6, 6, 10, 52.3]
}

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
for key, values in samples_results.items():
    ax.plot(n_samples, values, label=key)
ax.set_xlabel("Number of samples", fontsize=22, fontname="Source Code Pro")
ax.set_ylabel("Time (s)", fontsize=22, fontname="Source Code Pro")
fig.suptitle("Sampling 1,000 chains from a 4 components Gaussian mixture", fontsize=18, fontname="Source Code Pro")

ax.set_xscale('log')
ax.set_yscale('log')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.legend(frameon=False)

plt.savefig('samples.png', bbox_inches='tight', transparent=True)
