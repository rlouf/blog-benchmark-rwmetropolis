import argparse
from functools import partial
import time

import jax
import jax.numpy as np
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp


@partial(jax.jit, static_argnums=(1,))
def rw_metropolis_kernel(rng_key, logpdf, position, log_prob):
    """Moves the chains by one step using the Random Walk Metropolis algorithm.

    Attributes
    ----------
    rng_key: jax.random.PRNGKey
      Key for the pseudo random number generator.
    logpdf: function
      Returns the log-probability of the model given a position.
    position: np.ndarray, shape (n_dims,)
      The starting position.
    log_prob: float
      The log probability at the starting position.

    Returns
    -------
    Tuple
        The next positions of the chains along with their log probability.
    """
    key1, key2 = jax.random.split(rng_key)
    move_proposal = jax.random.normal(key1, shape=position.shape) * 0.1
    proposal = position + move_proposal
    proposal_log_prob = logpdf(proposal)

    log_uniform = np.log(jax.random.uniform(key2))
    do_accept = log_uniform < proposal_log_prob - log_prob

    position = np.where(do_accept, proposal, position)
    log_prob = np.where(do_accept, proposal_log_prob, log_prob)
    return position, log_prob


@partial(jax.jit, static_argnums=(1, 2))
def rw_metropolis_sampler(rng_key, n_samples, logpdf, initial_position):
    """Generate samples using the Random Walk Metropolis algorithm.

    Attributes
    ----------
    rng_key: jax.random.PRNGKey
        Key for the pseudo random number generator.
    n_samples: int
        Number of samples to generate per chain.
    logpdf: function
      Returns the log-probability of the model given a position.
    inital_position: np.ndarray (n_dims, n_chains)
      The starting position.

    Returns
    -------
    (n_samples, n_dim)
    """
    keys = jax.random.split(rng_key, n_samples)
    def mh_update(state, key):
      position, log_prob = state
      new_position, new_log_prob = rw_metropolis_kernel(key, logpdf, position, log_prob)
      return (new_position, new_log_prob), position
    logp = logpdf(initial_position)
    _, positions = jax.lax.scan(mh_update, (initial_position, logp), keys)
    return positions


def mixture_logpdf(x):
    """Log probability distribution function of a gaussian mixture model.

    Attribute
    ---------
    x: np.ndarray (4,)
        Position at which to evaluate the probability density function.

    Returns
    -------
    float
        The value of the log probability density function at x.
    """
    dist_1 = jax.partial(norm.logpdf, loc=-2.0, scale=1.2)
    dist_2 = jax.partial(norm.logpdf, loc=0, scale=1)
    dist_3 = jax.partial(norm.logpdf, loc=3.2, scale=5)
    dist_4 = jax.partial(norm.logpdf, loc=2.5, scale=2.8)
    log_probs = np.array([dist_1(x[0]), dist_2(x[1]), dist_3(x[2]), dist_4(x[3])])
    weights = np.array([0.2, 0.3, 0.1, 0.4])
    return -logsumexp(np.log(weights) - log_probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", default=1000, required=False, type=int, help="Number of samples to take"
    )
    parser.add_argument("--chains", default=4, type=int, help="Number of chains to run")
    parser.add_argument(
        "--precompiled", type=bool, default=False, help="Whether to time with a precompiled model (faster)"
    )
    args = parser.parse_args()

    n_dim = 4
    n_samples = args.samples
    n_chains = args.chains
    rng_key = jax.random.PRNGKey(42)

    rng_keys = jax.random.split(rng_key, n_chains)  # (nchains,)
    initial_position = np.zeros((n_dim, n_chains))  # (n_dim, n_chains)

    run_mcmc = jax.vmap(rw_metropolis_sampler, in_axes=(0, None, None, 1),
                        out_axes=0)
    positions = run_mcmc(rng_keys, n_samples, mixture_logpdf, initial_position)
    assert positions.shape == (n_chains, n_samples, n_dim)
    positions.block_until_ready()

    # TODO precompile=True logic
