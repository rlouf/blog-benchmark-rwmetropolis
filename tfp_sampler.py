import argparse
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def rw_metropolis_sampler(n_samples, n_chains, target):
    """Generate samples using the Random Walk Metropolis algorithm.

    Attributes
    ----------
    n_dims: int
        Number of dimensions of the distribution.
    n_samples: int
        Number of samples to generate per chain.
    n_chains: int
        The number of chains to generate.
    target: tfp distribution
        The distribution to sample from.

    Returns
    -------
    (n_samples, n_chains)
    """
    dtype = np.float32
    samples, _ = tfp.mcmc.sample_chain(
        num_results=n_samples,
        current_state=np.zeros(n_chains, dtype=dtype),
        kernel=tfp.mcmc.RandomWalkMetropolis(target.log_prob, seed=42),
        num_burnin_steps=0,
        parallel_iterations=8,
    )
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", default=1000, required=False, type=int, help="Number of samples to take"
    )
    parser.add_argument("--chains", default=4, type=int, help="Number of chains to run")
    parser.add_argument("--xla", default=False, type=bool, help="Use experimental XLA compilation")
    args = parser.parse_args()

    n_samples = args.samples
    n_chains = args.chains

    # Define the Gaussian Mixture model
    dtype = np.float32
    target = tfd.Mixture(
        cat=tfd.Categorical(probs=[0.2, 0.3, 0.1, 0.4]),
        components=[
            tfd.Normal(loc=dtype(-2.0), scale=dtype(1.2)),
            tfd.Normal(loc=dtype(0.0), scale=dtype(1.0)),
            tfd.Normal(loc=dtype(3.2), scale=dtype(5.0)),
            tfd.Normal(loc=dtype(2.5), scale=dtype(2.8)),
        ],
    )

    run_mcm = partial(rw_metropolis_sampler, n_samples, n_chains, target)
    if not args.xla:
        run_mcm()
    else:
        tf.xla.experimental.compile(run_mcm)
