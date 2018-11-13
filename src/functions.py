import timeit
import numpy as np
from numba import jit
import math
from matplotlib import pyplot as plt
import sys
import scipy # various algorithms
from matplotlib import pyplot as plt
from scipy.misc import logsumexp
import math
from scipy.stats import dirichlet


def simulate_traces(tau,dT,memory,trace_len,A,e,r,sigma,alpha):
    """

    :param tau: Inherent time scale of the sytem (seconds)
    :param dT:  Time resolution of experimental observations (seconds)
    :param memory: Dwell time on gene for Pol II molecules (units of tau)
    :param trace_len: Length of simulated trace (in units of dT)
    :param A: Transition probability matrix (KxK, where K is number of states)
    :param e: Emission probability matrix (always 3xK)
    :param r: Calibration term (AU/Pol II)
    :param sigma: Scale of Gaussian experimental noise
    :param alpha: MS2 rise time (in units of tau)
    :return:
    """
    # generate convolution kernel to deal with MS2 rise time
    if alpha > 0:
        alpha_vec = [(float(i + 1) / alpha + (float(i) / alpha)) / 2.0 * (i < alpha) * ((i + 1) <= alpha)
                     + ((alpha - i)*(1 + float(i) / alpha) / 2.0 + i + 1 - alpha) * (i < alpha) * (i + 1 > alpha)
                     + 1 * (i >= alpha) for i in range(memory)]

        #alpha_vec = np.array(alpha_vec[::-1])
    else:
        alpha_vec = np.array([1.0]*memory)
    # use this one to generate "realistic traces"
    kernel = np.ones(memory)*alpha_vec
    # generates traces assuming zero MS2 rise time--primarily as a  check
    kernel_unif = np.ones(memory)
    cv_factor = int(dT/tau) # cconversion factor giving number system steps per obs step
    # initialize lists and useful params
    K = A.shape[0]
    # track promoter states
    system_states = np.empty((trace_len*cv_factor), dtype='int')
    # track promoter initiation states
    initiation_states = np.empty((trace_len*cv_factor))
    # draw first system state
    system_states[0] = np.random.choice(K)
    # draw initiation state conditional on system state
    initiation_states[0] = np.random.choice(K, 1, p=e[:, system_states[0]])

    for i in range(1, trace_len*cv_factor):
        # time step
        system_states[i] = np.random.choice(K, 1, p=A[:, system_states[i-1]])
        initiation_states[i] = np.random.choice(K, 1, p=e[:, system_states[i]])


    # convolve to get fluo
    fluo_raw = np.convolve(kernel, initiation_states, mode='full')
    fluo_raw = fluo_raw[0:trace_len*cv_factor]*r
    fluo_unif = np.convolve(kernel_unif, initiation_states, mode='full')
    fluo_unif = fluo_unif[0:trace_len*cv_factor]*r

    # add noise
    noise_vec = np.random.randn(trace_len*cv_factor) * sigma
    fluo_noise = fluo_raw + noise_vec

    # output
    return fluo_noise, fluo_raw, fluo_unif, system_states, initiation_states

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


@jit
# forward filtering pass
def fwd_algorithm(init_vec, a_log, e_log, pi0_log):
    """
    :param init_vec: a single time series of initiation event counts
    :param a_log: current estimate of transition probability matrix (KxK)
    :param e_log: current estimate of initiation probability matrix (KxK)
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
        alpha_array[:, t] = logsumexp(a_log + prev, axis=1) + e_log[init_vec[t], :]
        prev = np.tile(alpha_array[:, t], (K, 1))
    return alpha_array


@jit
def bkwd_algorithm(init_vec, a_log, e_log, pi0_log):
    """
    :param init_vec: a single time series of initiation event counts
    :param a_log: current estimate of transition probability matrix (KxK)
    :param e_log: current estimate of initiation probability matrix (KxK)
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
        post = np.transpose(np.tile(beta_array[:, t+1] + e_log[init_vec[t+1], :], (K,1)))
        b_sums = post + a_log
        beta_array[:, t] = logsumexp(b_sums, axis=0)

    close_probs = [beta_array[l, 0] + pi0_log[l] + e_log[init_vec[0], l] for l in range(K)]
    return beta_array, logsumexp(close_probs)

# Helper function to calculate probability matrix probs for dirichlet prior
def dirichlet_prob(alpha_matrix,p_matrix):
    """

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
def log_prob_init(init_data, a_log, e_log, pi0_log, a_prior, e_prior):
    """
    :param init_data: list of lists. Each element is integer-valued time series indicating
            number of initiation events. Must be in [0, 1, 2]
    :param a_log: log of current estimate for transition prob matrix (KxK)
    :param e_log: log of current estimate for emission prob matrix (3xK)
    :param pi0_log: log of initial state PDF
    :param a_prior: matrix containing hyperparameters for transition prob prior (KxK)
    :param e_prior: matrix containing hyperparameters for emission prob prior 3xK)
    :return:
    """
    K = len(A[:, 0])
    # calculate param probs given priors
    a_prob = dirichlet_prob(alpha_matrix=a_prior, p_matrix=np.exp(a_log))
    e_prob = dirichlet_prob(alpha_matrix=e_prior, p_matrix=np.exp(e_log))
    # now calculate total prob for each sequence
    init_probs = []
    for i, init_vec in enumerate(init_data):
        fwd_matrix = fwd_algorithm(init_vec, a_log, e_log, pi0_log)
        init_probs.append(logsumexp(fwd_matrix[:, -1]))
    init_prob_total = np.sum(init_probs)
    # return toal probability
    return init_prob_total + a_prob + e_prob

def init_mcmc(init_data,a_prior,e_prior,n_steps=1000,burn_in=100):
    """
    :param init_data: list of lists. Each element is time series of initiation events
    :param a_prior: prior distribution over transition probability matrix (KxK)
    :param e_prior: prior distribution over initiation probability matrix (3xK)
    :param n_steps: number of MCMC steps
    :param burn_in: number of steps to discard (account for transient chain dynamics)
    :return: posterior distributions over transition probs, emission probs,
            system trajectories, and initiation trajectories
    """
    # N states
    K = len(e_prior[:, 0])
    # Initialize lists to store
    soft_arrays = []
    a_arrays = []
    e_arrays = []
    logL_list = []
    # Draw initial A and E values from priors
    a_init = np.empty(K)
    e_init = np.empty(K)
    for k in range(K):
        a_init[:, k] = dirichlet.rvs(a_prior[:, k])
        e_init[:, k] = dirichlet.rvs(e_prior[:, k])


if __name__ == "__main__":
    # memory
    memory = 7*10
    dT = 20
    tau = 2
    cv_factor = int(dT/tau)
    # Fix trace length for now
    trace_len = 100
    # Number of traces per batch
    sigma = 20
    r = 20
    alpha = 14
    A = np.array([[.9, .05, .1], [.05, .9, .1], [.05, .05, .8]])
    e = np.array([[.9, .3, .05], [.05, .6, .15], [.05, .1, .8]])

    a_prior = np.ones((3, 3))
    e_prior = np.ones((3, 3))

    test1 = np.random.randint(3, size=100)
    test2 = np.random.randint(3, size=100)
    pi0 = [1/3, 1/3, 1/3]
    start = timeit.timeit()
    lp = log_prob_init([test1, test2], np.log(A), np.log(e), np.log(pi0), a_prior, e_prior)
    stop = timeit.timeit()
    f_time = stop-start

    print(f_time)

    print(lp)

"""
    fluo_noise, fluo_raw, fluo_unif, system_states, initiation_states \
        = simulate_traces(tau,dT,memory,trace_len,A,e,r,sigma,alpha)

    fluo_obs = fluo_noise[np.arange(0,trace_len*cv_factor,cv_factor)]
    bins = range(50)
    plt.plot(fluo_noise)
    plt.plot(fluo_raw)
    plt.show()
"""