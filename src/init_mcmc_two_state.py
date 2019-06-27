import timeit
import numpy as np
from numba import jit
import math
from matplotlib import pyplot as plt
import sys
import scipy # various algorithms
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
from scipy.misc import logsumexp
import math
from scipy.stats import dirichlet
from scipy.stats import beta
from scipy.stats import multinomial
from sklearn.preprocessing import normalize


# forward filtering pass
def fwd_algorithm(init_vec, a_log, p_log, pi0_log):
    """
    :param init_vec: a single time series of initiation event counts
    :param a_log: current estimate of transition probability matrix (KxK)
    :param p_log: current estimate of initiation probability matrix (2xK)
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
        alpha_array[:, t] = logsumexp(a_log + prev, axis=1) + p_log[int(init_vec[t]), :]
        prev = np.tile(alpha_array[:, t], (K, 1))
    return alpha_array


#@jit
def bkwd_algorithm(init_vec, a_log, p_log, pi0_log):
    """
    :param init_vec: a single time series of initiation event counts
    :param a_log: current estimate of transition probability matrix (KxK)
    :param p_log: current estimate of initiation probability matrix (2xK)
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
    steps = np.arange(T-1)
    # Reverse direction
    steps = steps[::-1]
    for t in steps:
        post = np.transpose(np.tile(beta_array[:, t+1] + p_log[int(init_vec[t+1]), :], (K, 1)))
        b_sums = post + a_log
        beta_array[:, t] = logsumexp(b_sums, axis=0)
    close_probs = [beta_array[l, 0] + pi0_log[l] + p_log[int(init_vec[0]), l] for l in range(K)]
    return beta_array, logsumexp(close_probs)


#
#@jit
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
def empirical_counts(a_log, p_log, pi0_log, init_data, alpha_list, beta_list):
    """
    Returns counts of emission and transition events
    :param a_log: Float array (KxK). Log of current estimate of transition matrix
    :param p_log: Float array (2xK). Log of current estimate for emission matrix
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
    for t, init_vec in enumerate(init_data):
        alpha_array = alpha_list[t]
        beta_array = beta_list[t]
        seq_log_probs.append(logsumexp(alpha_array[:, -1]))
        seq_probs = alpha_array + beta_array
        seq_probs = seq_probs - logsumexp(seq_probs, axis=0)
        full_seq_probs.append(seq_probs)

    # Now calculate effective number of transitions (empirical transition matrix)
    a_log_empirical = np.zeros_like(a_log) - np.Inf
    K = len(a_log[:, 0])
    # transition counts
    for k in range(K):
        for l in range(K):
            # store log probs for each sequence
            a_probs = []
            # current transition prob from k to l
            akl = a_log[l, k]
            for i, init_vec in enumerate(init_data):
                T = len(init_vec)
                a = alpha_list[i]
                b = beta_list[i]
                event_list = [a[k, t] + b[l, t + 1] + akl + p_log[int(init_vec[t + 1]), l] for t in range(T - 1)]
                a_probs.append(logsumexp(event_list) - seq_log_probs[i]) # NL: why do this?

            a_log_empirical[l, k] = logsumexp(a_probs)

    p_empirical = np.zeros_like(p_log)
    for i, init_vec in enumerate(init_data):
        seq_probs = np.exp(full_seq_probs[i])
        for t in range(len(init_vec)):
            p_empirical[int(init_vec[t]), :] += seq_probs[:, t]

    pi0_log_empirical = np.zeros_like(pi0_log)
    for i, init_vec in enumerate(init_data):
        seq_probs = full_seq_probs[i]
        pi0_log_empirical += seq_probs[:, 0]

    return a_log_empirical, np.log(p_empirical), pi0_log_empirical, seq_log_probs, full_seq_probs

def log_prob_init(init_data, a_mat, p_mat, pi0_vec):
    """
    :param init_data: list of lists. Each element is integer-valued time series indicating
            number of initiation events. Must be in [0, 1, 2]
    :param a_mat: current estimate for transition prob matrix (KxK)
    :param p_vec: current estimate for emission prob matrix (2xK)
    :param pi0_vec: initial state PDF
    :param a_prior: matrix containing hyperparameters for transition prob prior (KxK)
    :param e_prior: matrix containing hyperparameters for emission prob prior 2xK)
    :param pi0_prior: vector containing hyperparameters for initial state prob prior 1xK)
    :return: "total_prob": total probability of sequence, "fwd_list": list of forward matrices
    """
    # calculate total prob for each sequence
    init_probs = []
    fwd_list = []
    for i, init_vec in enumerate(init_data):
        fwd_matrix = fwd_algorithm(init_vec, np.log(a_mat), np.log(p_mat), np.log(pi0_vec))
        fwd_list.append(fwd_matrix)
        init_probs.append(logsumexp(fwd_matrix[:, -1]))
    init_prob_total = np.sum(init_probs)
    total_prob = init_prob_total
    # return total probability
    return total_prob, fwd_list


def init_single_prom_mcmc(init_data, a_prior, p_prior, pi0_prior, n_steps=1000, burn_in=100):

    """
    :param init_data: list of lists. Each element is time series of initiation events
    :param a_prior: prior distribution over transition probability matrix (KxK)    
    :param p_prior: prior distribution over initiation probability vector (2xK)    
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
    p_array_list = []
    pi0_array_list = []
    logL_list = []
    full_seq_probs_list = []
    tr_count_list = []
    emission_count_list = []
    init_count_list = []
    # Draw initial A and E values from priors
    a_init = np.empty((K, K))
    p_init = np.empty((2, K))
    for k in range(K):
        a_init[:, k] = dirichlet.rvs(a_prior[:, k])
        p1 = np.random.beta(p_prior[0, k], p_prior[1, k])
        p_init[:, k] = [p1, 1-p1]
    pi0_init = dirichlet.rvs(pi0_prior)
    pi0_init = pi0_init[0, :]

    ############################# Initialization Step #############################

    # calculate initial model probability
    total_prob, alpha_list = log_prob_init(init_data, a_init, p_init, pi0_init)

    # Calculate backward probabilities for each sequence
    beta_list = []
    for t, init_vec in enumerate(init_data):
        beta_array, cp = bkwd_algorithm(init_vec, np.log(a_init), np.log(p_init), np.log(pi0_init))
        beta_list.append(beta_array)

    # Calculate empirical emission and transition probability matricies
    a_log_empirical, p_log_empirical, pi0_log_empirical, seq_log_probs, full_seq_probs = \
        empirical_counts(np.log(a_init), np.log(p_init), np.log(pi0_init), init_data, alpha_list, beta_list)

    # record first iteration values
    logL_list.append(total_prob)
    a_array_list.append(a_init)
    p_array_list.append(p_init)
    pi0_array_list.append(pi0_init)
    full_seq_probs_list.append(full_seq_probs)
    tr_count_list.append(a_log_empirical)
    emission_count_list.append(p_log_empirical)
    init_count_list.append(pi0_log_empirical)
    # now iterate
    for sample in range(n_steps):
        # initialize current values for variables
        a_log_empirical = tr_count_list[sample-1]
        p_log_empirical = emission_count_list[sample - 1]
        pi0_log_empirical = init_count_list[sample - 1]

        # Initialize empty emission and transition matrices
        a_current = np.empty_like(a_prior)
        p_current = np.empty_like(p_prior)

        # Sample new initial PDF vector
        pi0_parameters = pi0_prior + np.exp(pi0_log_empirical)
        pi0_current = dirichlet.rvs(pi0_parameters)[0]

        # Sample new emission probability matrix
        p_parameters = p_prior + np.exp(p_log_empirical)
        for k in range(K):
            p_current[:, k] = dirichlet.rvs(p_parameters[:, k])

        # enforce ascending order in production rates
        #idx = np.argsort(p_current[1, :])
        #p_current = p_current[:, idx]

        """
        if sample == 150:
            print(p_parameters)
            print(p_current)
            print(p_array_list[sample-1])
            sys.exit(0)
            """
        # Sample new transition probability matrix
        a_parameters = a_prior + np.exp(a_log_empirical)
        for k in range(K):
            a_current[:, k] = dirichlet.rvs(a_parameters[:, k])
        # enforce sort ordering
        #a_temp = a_current[idx,:]
        #a_current = a_temp[:, idx]
        # Calculate current probability
        prob_curr, alpha_list = log_prob_init(init_data, a_current, p_current, pi0_current)

        # Filter sequence using fwd-bkwd algorithm to obtain  step-by-step state probabilities
        beta_list = []
        full_seq_probs = []
        seq_log_probs = []
        for t, init_vec in enumerate(init_data):
            beta_array, cp = bkwd_algorithm(init_vec, np.log(a_current), np.log(p_current), np.log(pi0_current))
            beta_list.append(beta_array)

        a_log_empirical, e_log_empirical, pi0_log_empirical, seq_log_probs, full_seq_probs = \
            empirical_counts(np.log(a_current), np.log(p_current), np.log(pi0_current), init_data, alpha_list, beta_list)
        # record updates
        p_array_list.append(p_current)
        a_array_list.append(a_current)
        pi0_array_list.append(pi0_current)
        full_seq_probs_list.append(full_seq_probs)
        tr_count_list.append(a_log_empirical)
        emission_count_list.append(e_log_empirical)
        init_count_list.append(pi0_log_empirical)
        logL_list.append(prob_curr)


         #   print('step:')
         #   print(sample)
         #   print(prob_curr)

    return logL_list, p_array_list, a_array_list, pi0_array_list

