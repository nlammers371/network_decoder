import timeit
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import dirichlet
from mcmc_helper_functions import *
from numpy.random import gamma

def init_mcmc(init_data, a_prior, lambda_priors, pi0_prior, n_steps=1000, burn_in=100):


    """
    :param init_data: list of lists. Each element is time series of initiation events
    :param a_prior: prior distribution over transition probability matrix (KxK)
    :param lambda_priors: alpha and beta coefficients dictating shape of prior (gamma) distributions over Poisson rate values(2xK)
    :param pi0_prior: prior distribution over initial state PDF
    :param n_steps: number of MCMC steps
    :param burn_in: number of steps to discard (account for transient chain dynamics)
    :return: posterior distributions over transition probs, emission probs,
            system traject  ories, and initiation trajectories
    """
    # N states
    K = len(a_prior[:, 0])
    # Initialize lists to store
    a_array_stack = np.empty((K, K, n_steps))
    lambda_array = np.empty((K, n_steps))
    pi0_array = np.empty((K, n_steps))
    logL_vec = np.empty(n_steps)
    full_seq_probs_list = []
    tr_count_array = np.empty((K, K, n_steps))
    emission_count_array = np.empty((K, n_steps))
    init_count_array = np.empty((K, n_steps))
    visit_count_array = np.empty((K, n_steps))
    # Draw initial A
    a_init = np.empty((K, K))
    for k in range(K):
        a_init[:, k] = dirichlet.rvs(a_prior[:, k])
    # lambda values
    lambda_vec_init = np.random.gamma(lambda_priors[0, :], 1/lambda_priors[1, :])

    pi0_init = dirichlet.rvs(pi0_prior)
    pi0_init = pi0_init[0, :]

    # ---------------------------------------------- Initialization Step --------------------------------------------- #

    # calculate initial model probability
    total_prob, alpha_list = log_prob_init(init_data, a_init, lambda_vec_init, pi0_init, a_prior, lambda_priors, pi0_prior)

    # Calculate backward probabilities for each sequence
    beta_list = []
    for t, init_vec in enumerate(init_data):
        beta_array, cp = bkwd_algorithm(init_vec, np.log(a_init), lambda_vec_init, np.log(pi0_init))
        beta_list.append(beta_array)

    # Calculate empirical emission and transition probability matricies
    a_counts, lambda_counts, pi0_counts, seq_log_probs, full_seq_probs, state_visits = \
        empirical_counts(np.log(a_init), lambda_vec_init, np.log(pi0_init), init_data, alpha_list, beta_list)

    # record first iteration values
    logL_vec[0] = total_prob
    a_array_stack[:, :, 0] = a_init
    lambda_array[:, 0] = lambda_vec_init
    pi0_array[:, 0] = pi0_init
    full_seq_probs_list.append(full_seq_probs)
    tr_count_array[:, :, 0] = a_counts
    emission_count_array[:, 0] = lambda_counts
    init_count_array[:, 0] = pi0_counts
    visit_count_array[:, 0] = state_visits
    # now iterate
    for sample in range(1, n_steps):
        # initialize current values for variables
        a_counts = tr_count_array[:, :, sample-1]
        lambda_counts = emission_count_array[:, sample - 1]
        state_visits = visit_count_array[:, sample-1]
        pi0_counts = init_count_array[:, sample - 1]

        # Initialize empty emission and transition matrices
        a_current = np.empty_like(a_prior)        

        # Sample new initial PDF vector
        pi0_parameters = pi0_prior + pi0_counts
        pi0_current = dirichlet.rvs(pi0_parameters)[0]

        # Sample new emission probability matrix
        alpha_params = lambda_priors[0, :] + lambda_counts
        theta_params = 1 / (lambda_priors[1, :] + state_visits)
        lambda_vec_current = gamma(alpha_params, theta_params)
        # Sample new transition probability matrix
        a_parameters = a_prior + a_counts/np.sum(a_counts) * 500
        for k in range(K):
            a_current[:, k] = dirichlet.rvs(a_parameters[:, k])

        # Calculate current probability
        prob_curr, alpha_list = log_prob_init(init_data, a_current, lambda_vec_current, pi0_current, a_prior, lambda_priors, pi0_prior)
        # Filter sequence using fwd-bkwd algorithm to obtain step-by-step state probabilities
        beta_list = []
        for t, init_vec in enumerate(init_data):
            beta_array, cp = bkwd_algorithm(init_vec, np.log(a_current), lambda_vec_current, np.log(pi0_current))
            beta_list.append(beta_array)

        a_counts, lambda_counts, pi0_log_empirical, seq_log_probs, full_seq_probs, state_visits = \
            empirical_counts(np.log(a_current), lambda_vec_current, np.log(pi0_current), init_data, alpha_list, beta_list)

        # record updates
        lambda_array[:, sample] = lambda_vec_current
        a_array_stack[:, :, sample] = a_current
        pi0_array[:, sample] = pi0_current
        full_seq_probs_list.append(full_seq_probs)
        tr_count_array[:, :, sample] = a_counts
        emission_count_array[:, sample] = lambda_counts
        init_count_array[:, sample] = pi0_log_empirical
        logL_vec[sample] = prob_curr
        visit_count_array[:, sample] = state_visits

    return logL_vec, lambda_array, a_array_stack, pi0_array

#if __name__ == "__main__":
