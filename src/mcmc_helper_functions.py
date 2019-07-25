import time
import numpy as np
from numba import jit
import math
from matplotlib import pyplot as plt
import sys
import scipy  # various algorithms
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
from scipy.special import factorial
from scipy.stats import dirichlet
from scipy.stats import poisson

from scipy.stats import multinomial
from sklearn.preprocessing import normalize

"""
#Helper function to calculate log likelihood of a proposed state
def log_L_fluo(fluo, v, state, noise):

    :param fluo: Fluorescence value
    :param v: emission vector
    :param state: state
    :param noise: standard deviation of emissions
    :return: log likelihood associated with fluorescence

    noise_lambda = noise ** -2
    logL = 0.5 * math.log(noise_lambda) - 0.5 * np.log(2*np.pi) - 0.5 * noise_lambda * (fluo - v[state])**2
    return logL
"""


# @jit
# forward filtering pass
def fwd_algorithm(init_vec, a_log, lambda_vec, pi0_log):
    """
    :param init_vec: a single time series of initiation event counts
    :param a_log: current estimate of transition probability matrix (KxK)
    :param lambda_vec: current estimate of (Poisson) initiation rates (1xK)
    :param pi0_log: initial state PDF (1xK)
    :return: K x T vector of  log probabilities
    """
    K = len(a_log[0, :])
    T = len(init_vec)

    # Allocate alpha array to store log probs
    alpha_array = np.zeros((K, T), dtype=float) - np.Inf
    # Iterate through time points
    prev = np.tile(pi0_log, (K, 1))
    for t in range(0, T):
        init_long = np.tile(init_vec[t], (1, K))
        lambda_log = -lambda_vec + np.multiply(init_long, np.log(lambda_vec)) - np.log(factorial(init_long))
        alpha_array[:, t] = np.logaddexp.reduce(a_log + prev, axis=1) + lambda_log
        prev = np.tile(alpha_array[:, t], (K, 1))
    return alpha_array

# @jit
def bkwd_algorithm(init_vec, a_log, lambda_vec, pi0_log):
    """
    :param init_vec: a single time series of initiation event counts
    :param a_log: current estimate of transition probability matrix (KxK)
    :param lambda_vec: current estimate of (Poisson) initiation rates (1xK)
    :param pi0_log: initial state PDF (Kx1)
    :return: K x T vector of  log probabilities
    """
    K = len(a_log[0, :])
    T = len(init_vec)
    # Allocate alpha array to store log probs
    beta_array = np.zeros((K, T), dtype=float) - np.Inf
    # initialize--We basically ignore this step
    beta_array[:, -1] = np.log(1.0)
    # Iteration
    steps = np.arange(T - 1)
    # Reverse direction
    steps = steps[::-1]
    for t in steps:
        init_long = np.tile(init_vec[t+1], K)
        lambda_log = -lambda_vec + np.multiply(init_long, np.log(lambda_vec)) - np.log(factorial(init_long))
        post = np.transpose(np.tile(beta_array[:, t + 1] + np.transpose(lambda_log), (K, 1)))
        b_sums = post + a_log
        beta_array[:, t] = np.logaddexp.reduce(b_sums, axis=0)
    # calculate close probs
    lambda_log = [np.log(poisson.pmf(init_vec[0], mu=lb)) for lb in lambda_vec]
    close_probs = [beta_array[l, 0] + pi0_log[l] + np.log(poisson.pmf(init_vec[0], mu=lambda_vec[l])) for l in range(K)]
    return beta_array, np.logaddexp.reduce(close_probs)

def particle_filter(init_vec, a_mat, e_mat, pi0_vec, n_particles):
    """
    Simulates trajectories of a colleciton of particles given specified system parameters
    :param init_vec: time series of observed initiation events
    :param a_mat: transition probability matrix
    :param e_mat: initiation probability matrix
    :param pi0_vec: pdf over initial states
    :param n_particles: number of perticles to simulate
    :return: particle_array (KxT array of particle trajectories) ; a_counts (KxK), e_counts (3xK), p_counts
    """
    K = len(a_mat[0, :])
    T = len(init_vec)

    # Allocate particle array to store particle counts
    forward_array = np.zeros((K, T+1), dtype=float)
    # array to track transition events
    a_counts = np.zeros_like(a_mat)
    # array to count initiation events
    e_counts = np.zeros_like(e_mat)
    # sample initial state
    forward_array[:, 0] = np.random.multinomial(n_particles, pi0_vec)
    # Iterate through time points
    for t in range(1, T+1):
        # first naively sample state transitions
        e_vec = e_mat[init_vec[t-1], :]
        a_temp = np.zeros_like(a_mat)
        for k in range(0, K):
            p_vec = np.multiply(e_vec, a_mat[:, k])
            a_temp[:, k] += np.random.multinomial(forward_array[k, t-1], p_vec)
        # record
        forward_array[:, t] = np.sum(a_temp, axis=1)
        e_counts[init_vec[t-1], :] += np.sum(a_temp, axis=1)
        a_counts += a_temp
    # return arrays
    return forward_array[:, 1:], a_counts, e_counts, forward_array[:, 0]

#
# @jit
def dirichlet_prob(alpha_matrix, p_matrix):
    """
    Helper function to calculate probability matrix probs for dirichlet prior
    :param alpha_matrix: matrix containing dirichlet hyperparameters.
            1-to-1 correspondence with p_matrix elements
    :param p_matrix: probability matrix. Assume each column is independent, normalized distribution
    :return: total log likelihood of p_matrix given alpha params
    """
    logL = 0
    n_col = len(p_matrix[0, :])
    for i in range(n_col):
        logL += dirichlet.logpdf(p_matrix[:, i], alpha_matrix[:, i])

    return logL


# function to calculate the probability of a parameter set given priors and data
def log_prob_init(init_data, a_mat, lambda_vec, pi0_vec, a_prior, e_prior, pi0_prior):
    """
    :param init_data: list of lists. Each element is integer-valued time series indicating
            number of initiation events. Must be in [0, 1, 2]
    :param a_mat: current estimate for transition prob matrix (KxK)
    :param lambda_vec: current estimate for emission rates (1xK)
    :param pi0_vec: initial state PDF
    :param a_prior: matrix containing hyperparameters for transition prob prior (KxK)
    :param lambda_prior: matrix containing hyperparameters for emission prob prior 3xK)
    :param pi0_prior: vector containing hyperparameters for initial state prob prior 1xK)
    :return: "total_prob": total probability of sequence, "fwd_list": list of forward matrices
    """
    # calculate total prob for each sequence
    init_probs = []
    fwd_list = []
    for i, init_vec in enumerate(init_data):
        fwd_matrix = fwd_algorithm(init_vec, np.log(a_mat), lambda_vec, np.log(pi0_vec))
        fwd_list.append(fwd_matrix)
        init_probs.append(np.logaddexp.reduce(fwd_matrix[:, -1]))
    init_prob_total = np.sum(init_probs)
    total_prob = init_prob_total  # + a_prob + e_prob + pi0_prob
    # return total probability
    return total_prob, fwd_list


def empirical_counts(a_log, lambda_vec, pi0_log, init_data, alpha_list, beta_list):
    """
    Returns counts of emission and transition events
    :param a_log: Float array (KxK). Log of current estimate of transition matrix
    :param lambda_vec: Float vec (1xK). current estimate for Poisson initiation rates
    :param pi0_log: Float array (1xK). Log of current estimate for inital state PDF
    :param init_data: List of lists. Each item is a time series of initiation events
    :param alpha_list: List of lists. Each element is array of fwd probabilities
    :param beta_list: List of lists. Each element is array of bkwd probabilities
    :return: empirical counts for transition events, initiation events, first state PD
            list of marginal hidden state probabilities, list of total sequence probabilities
    """
    # Calculate total marginal probability for each z_it
    seq_log_probs = []
    full_seq_probs = []
    visit_counts = np.zeros((1, len(pi0_log)))
    for t in range(len(init_data)):
        alpha_array = alpha_list[t]
        beta_array = beta_list[t]
        seq_log_probs.append(np.logaddexp.reduce(alpha_array[:, -1]))
        seq_probs = alpha_array + beta_array
        seq_probs = seq_probs - np.logaddexp.reduce(seq_probs, axis=0)
        full_seq_probs.append(seq_probs)
        # calculate total visits per state
        visit_counts += np.sum(np.exp(seq_probs), axis=1)
    # Now calculate effective number of transitions (empirical transition matrix)
    a_log_empirical = np.zeros_like(a_log) - np.Inf
    K = len(lambda_vec)
    # transition counts
    for k in range(K):
        for l in range(K):
            # store log probs for each sequence
            a_probs = []
            # current transition prob from k to l
            akl = a_log[l, k]
            e_rate = lambda_vec[l]
            for i, init_vec in enumerate(init_data):
                T = len(init_vec)
                a = alpha_list[i]
                b = beta_list[i]
                event_list = [a[k, t] + b[l, t + 1] + akl + -e_rate + np.multiply(init_vec[t + 1], np.log(e_rate)) -
                              np.log(factorial(init_vec[t + 1])) for t in range(T - 1)]
                a_probs.append(np.logaddexp.reduce(event_list) - seq_log_probs[i])  # NL: why do this?

            a_log_empirical[l, k] = np.logaddexp.reduce(a_probs)

    lambda_counts = np.zeros_like(lambda_vec)
    for i, init_vec in enumerate(init_data):
        seq_probs = np.exp(full_seq_probs[i])
        for t in range(len(init_vec)):
            lambda_counts += seq_probs[:, t]*init_vec[t]

    pi0_log_empirical = np.zeros_like(pi0_log)
    for i, init_vec in enumerate(init_data):
        seq_probs = full_seq_probs[i]
        pi0_log_empirical += seq_probs[:, 0]

    return np.exp(a_log_empirical), lambda_counts, np.exp(pi0_log_empirical), seq_log_probs, full_seq_probs, visit_counts[0]