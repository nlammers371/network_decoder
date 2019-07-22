import timeit
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import dirichlet
from mcmc_helper_functions import *


def init_mcmc(init_data, a_prior, e_prior, pi0_prior, n_steps=1000, burn_in=100):


    """
    :param init_data: list of lists. Each element is time series of initiation events
    :param a_prior: prior distribution over transition probability matrix (KxK)
    :param a_jump: standard deviation of clipped gaussian used to generate a proposals
    :param e_prior: alpha and beta coefficients dictating shape of prior Beta distriutions (2xK)
    :param e_jump: standard deviation of clipped gaussian used to generate e proposals
    :param pi0_prior: prior distribution over initial state PDF
    :param n_steps: number of MCMC steps
    :param burn_in: number of steps to discard (account for transient chain dynamics)
    :return: posterior distributions over transition probs, emission probs,
            system traject  ories, and initiation trajectories
    """
    # N states
    K = len(a_prior[:, 0])
    # Initialize lists to store
    soft_arrays = []
    a_array_list = []
    e_array_list = []
    pi0_array_list = []
    logL_list = []
    full_seq_probs_list = []
    tr_count_list = []
    emission_count_list = []
    init_count_list = []
    # Draw initial A and E values from priors
    a_init = np.empty((K, K))
    e_array_init = np.empty_like(e_prior)
    for k in range(K):
        a_init[:, k] = dirichlet.rvs(a_prior[:, k])        
        e_array_init[:, k] = dirichlet.rvs(e_prior[:, k])
        
    pi0_init = dirichlet.rvs(pi0_prior)
    pi0_init = pi0_init[0, :]
    ############################# Initialization Step #############################

    # calculate initial model probability
    total_prob, alpha_list = log_prob_init(init_data, a_init, e_array_init, pi0_init, a_prior, e_prior, pi0_prior)

    # Calculate backward probabilities for each sequence
    beta_list = []
    for t, init_vec in enumerate(init_data):
        beta_array, cp = bkwd_algorithm(init_vec, np.log(a_init), np.log(e_array_init), np.log(pi0_init))
        beta_list.append(beta_array)

    # Calculate empirical emission and transition probability matricies
    a_log_empirical, e_log_empirical, pi0_log_empirical, seq_log_probs, full_seq_probs = \
        empirical_counts(np.log(a_init), np.log(e_array_init), np.log(pi0_init), init_data, alpha_list, beta_list)

    # record first iteration values
    logL_list.append(total_prob)
    a_array_list.append(a_init)
    e_array_list.append(e_array_init)
    pi0_array_list.append(pi0_init)
    full_seq_probs_list.append(full_seq_probs)
    tr_count_list.append(a_log_empirical)
    emission_count_list.append(e_log_empirical)
    init_count_list.append(pi0_log_empirical)

    # now iterate
    for sample in range(n_steps):
        # initialize current values for variables
        a_log_empirical = tr_count_list[sample-1]
        e_empirical = np.exp(emission_count_list[sample - 1])
        pi0_log_empirical = init_count_list[sample - 1]

        # Initialize empty emission and transition matrices
        a_current = np.empty_like(a_prior)
        e_array_current = np.empty_like(e_array_init)

        # Sample new initial PDF vector
        pi0_parameters = pi0_prior + np.exp(pi0_log_empirical)
        pi0_current = dirichlet.rvs(pi0_parameters)[0]

        # Sample new emission probability matrix
        e_parameters = e_prior + e_empirical
        for k in range(K):
            e_array_current[:, k] = dirichlet.rvs(e_parameters[:, k])
        # Sample new transition probability matrix
        a_parameters = a_prior + np.exp(a_log_empirical)
        for k in range(K):
            a_current[:, k] = dirichlet.rvs(a_parameters[:, k])

        # Calculate current probability
        prob_curr, alpha_list = log_prob_init(init_data, a_current, e_array_current, pi0_current, a_prior, e_prior, pi0_prior)
        # Filter sequence using fwd-bkwd algorithm to obtain step-by-step state probabilities
        beta_list = []
        for t, init_vec in enumerate(init_data):
            beta_array, cp = bkwd_algorithm(init_vec, np.log(a_current), np.log(e_array_current), np.log(pi0_current))
            beta_list.append(beta_array)

        a_log_empirical, e_log_empirical, pi0_log_empirical, seq_log_probs, full_seq_probs = \
            empirical_counts(np.log(a_current), np.log(e_array_current), np.log(pi0_current), init_data, alpha_list, beta_list)

        # record updates
        e_array_list.append(e_array_current)
        a_array_list.append(a_current)
        pi0_array_list.append(pi0_current)
        full_seq_probs_list.append(full_seq_probs)
        tr_count_list.append(a_log_empirical)
        emission_count_list.append(e_log_empirical)
        init_count_list.append(pi0_log_empirical)
        logL_list.append(prob_curr)

    return logL_list, e_array_list, a_array_list, pi0_array_list

if __name__ == "__main__":
    # memory
    memory = 7*10
    dT = 20
    tau = 2
    cv_factor = int(dT/tau)
    # Fix trace length for now
    trace_len = 2000
    # Number of traces per batch
    sigma = 20
    r = 20
    alpha = 14
    # set true parameters to be inferred
    A = np.array([[.9, .05, .1],
                  [.05, .9, .1],
                  [.05, .05, .8]])
    e = np.array([[.9, .3, .05],
                  [.05, .6, .15],
                  [.05, .1, .8]])

    # set priors
    a_prior = np.ones((3, 3))
    e_prior = np.ones((3, 3))
    pi0_prior = np.ones((3))
    # simulate data
    fluo_noise, fluo_raw, fluo_unif, system_states, initiation_states \
        = simulate_traces(tau, dT, memory, trace_len, A, e, r, sigma, alpha)

    start = timeit.default_timer()
    n_steps = 100
    logL_list, e_array_list, a_array_list, pi0_array_list = \
        init_mcmc([initiation_states], a_prior, e_prior, pi0_prior, n_steps=n_steps, burn_in=1)
    stop = timeit.default_timer()

    print(stop-start)
    print(a_array_list[0])
    print(a_array_list[-1])
    print(e_array_list[0])
    print(e_array_list[-1])

    plt.plot(logL_list)
    plt.show()

"""
    fluo_noise, fluo_raw, fluo_unif, system_states, initiation_states \
        = simulate_traces(tau,dT,memory,trace_len,A,e,r,sigma,alpha)

    fluo_obs = fluo_noise[np.arange(0,trace_len*cv_factor,cv_factor)]
    bins = range(50)
    plt.plot(fluo_noise)
    plt.plot(fluo_raw)
    plt.show()
"""