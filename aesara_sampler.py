import argparse

from aeppl import joint_logprob
import aesara
import aesara.tensor as at
import numpy as np


def rw_metropolis_kernel(srng, logprob_fn, num_chains):
    """Build the random walk Rosenbluth-Metropolis-Hastings (RNH) kernel."""

    def one_step(position, logprob):
        """Generate one sample using the random walk RMH algorithm.

        Attributes
        ----------
        position:
            The initial position.
        logprob:
            The initial value of the logprobability.

        Returns
        ------
        The next positions and values of the logprobability.

        """
        move_proposal = 0.1 * srng.normal(0, 1, size=num_chains)
        proposal = position + move_proposal
        proposal_logprob = logprob_fn(proposal)

        log_uniform = at.log(srng.uniform(size=num_chains))
        do_accept = log_uniform < proposal_logprob - logprob

        position = at.where(do_accept, proposal, position)
        logprob = at.where(do_accept, proposal_logprob, logprob)

        return position, logprob

    return one_step


def rw_metropolis_sampler(srng, logprob_fn, init_position, num_samples, n_chains):
    """Build the random walk metropolis sampler."""

    init_logprob = logprob_fn(init_position)
    kernel = rw_metropolis_kernel(srng, logprob_fn, n_chains)
    results, updates = aesara.scan(
        kernel,
        outputs_info=(init_position, init_logprob),
        n_steps=num_samples,
    )

    return results, updates


def build_logpdf(srng, num_chains):

    def mixture_logpdf(y_vv):
        loc = np.array([[-2, 0, 3.2, 2.5]]).T
        scale = np.array([[1.2, 1, 5, 2.8]]).T
        weights = np.array([0.2, 0.3, 0.1, 0.4])

        N_rv = srng.normal(loc, scale, size=(4, num_chains))
        I_rv = srng.categorical(weights)
        Y_rv = N_rv[I_rv]

        logprob = []
        for i in range(4):
            i_vv = at.as_tensor(i, dtype="int64")
            logprob.append(joint_logprob({Y_rv: y_vv, I_rv: i_vv}, sum=False))
        logprob = at.stack(logprob, axis=0)

        return at.logsumexp(at.log(weights.reshape(-1, 1)) + logprob, axis=0)

    return mixture_logpdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", default=1000, required=False, type=int, help="Number of samples to take"
    )
    parser.add_argument("--chains", default=4, type=int, help="Number of chains to run")
    args = parser.parse_args()

    n_samples = args.samples
    n_chains = args.chains

    srng = at.random.RandomStream()
    mixture_logpdf = build_logpdf(srng, n_chains)

    n_samples_at = at.iscalar()
    initial_position_at = at.vector()
    results, updates = rw_metropolis_sampler(srng, mixture_logpdf, initial_position_at, n_samples_at, n_chains)

    sampling_fn = aesara.function((initial_position_at, n_samples_at), results, updates=updates)
    samples, _ = sampling_fn(np.zeros(n_chains), n_samples)
    assert samples.shape == (n_samples, n_chains)
