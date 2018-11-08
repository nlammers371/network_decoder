import numpy as np
import math
from matplotlib import pyplot as plt
import sys
import scipy # various algorithms
from matplotlib import pyplot as plt
from scipy.misc import logsumexp
import math
from itertools import chain


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
    print(system_states[0])
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
# forward filtering pass
def fwd_algorithm(init_vec, A_log, E_log, pi0_log):
    """
    :param init_vec: a single time series of initiation event counts
    :param A_log: current estimate of transition probability matrix (KxK)
    :param E_log: current estimate of initiation probability matrix (KxK)
    :param pi0_log: initial state PDF (Kx1)
    :return: K x T vector of  log probabilities
    """
    K = len(A_log[0.:])
    T = len(init_vec)
    # Allocate alpha array to store log probs
    alpha_array = np.zeros((K, T), dtype=float) - np.Inf
    # Iterate through time points
    prev = np.tile(pi0_log, (1, K))
    for t in range(0, T):
        alpha_array[:, t] = logsumexp(A_log + prev,axis=1) + E_log[init_vec[t]]
        prev = np.tile(alpha_array[:, t], (1, K))
    return alpha_array


def bkwd_algorithm(init_vec, A_log, E_log, pi0_log):
    """
    :param init_vec: a single time series of initiation event counts
    :param A_log: current estimate of transition probability matrix (KxK)
    :param E_log: current estimate of initiation probability matrix (KxK)
    :param pi0_log: initial state PDF (Kx1)
    :return: K x T vector of  log probabilities
    """
    K = len(A_log[0.:])
    T = len(init_vec)
    # Allocate alpha array to store log probs
    beta_array = np.zeros((K, T), dtype=float) - np.Inf
    # initialize--We basically ignore this step
    beta_array[:, -1] = np.log(1.0)
    # Iteration
    steps = np.arange(T-1)
    steps = steps[::-1]
    for t in steps:
        post = beta_array[:, t + 1]
        for k in range(K):
            # get logL of transitions into state k from each of the previous states
            b_sums = [A_log[l, k] + post[l] + log_L_fluo(fluo_vec[t+1],v,l,noise) for l in xrange(K)]
            # scipy has a built in function for handling logsums
            beta_array[k, t] = logsumexp(b_sums)
        if t == 0:
            close_probs = [beta_array[l,0] + pi0_log[l] + log_L_fluo(fluo_vec[t],v,l,noise) for l in xrange(K)]
    return beta_array, logsumexp(close_probs)

#Function to calculate likelhood of data given estimated parameters
def log_likelihood(fluo, A_log, v, noise, pi0_log, alpha, beta):
    """

    :param fluo: List of fluo time series
    :param A_log: Log of transition matrix
    :param v: Emission States
    :param noise: Noise (stddev of signal)
    :param pi0_log: Log of initial state PDF
    :param alpha: Forward matrix
    :param beta: Backward matrix
    :return: Log Probability
    """
    l_score = 0
    K = len(v)
    for f, fluo_vec in enumerate(fluo):
        # Get log likelihood of sequence
        p_x = logsumexp(alpha[f][:,-1])
        for t in xrange(len(fluo_vec)):
            for k in xrange(K):
                #Likelihood of observing F(t)
                l_score += math.exp(alpha[f][k,t] + beta[f][k,t] - p_x) * log_L_fluo(fluo_vec[t], v, k, noise)
            if t == 0:
                #Likelihood of sequencce starting with k
                for k in xrange(K):
                    l_score += math.exp(alpha[f][k,t] + beta[f][k,t] - p_x) * (pi0_log[k] + alpha[f][k,t] + beta[f][k,t])
            else:
                #Likelihood of transition TO l FROM k
                for k in xrange(K):
                    for l in xrange(K):
                        l_score += math.exp(alpha[f][l,t] + beta[f][l,t] + alpha[f][k,t-1] + beta[f][k,t-1] - p_x) * A_log[l,k]

    return l_score


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
    A = np.array([[.9,.05,.1],[.05,.9,.1],[.05,.05,.8]])
    e = np.array([[.95, .3, .05], [.05, .6, .15], [.00, .1, .8]])

    fluo_noise, fluo_raw, fluo_unif, system_states, initiation_states \
        = simulate_traces(tau,dT,memory,trace_len,A,e,r,sigma,alpha)

    fluo_obs = fluo_noise[np.arange(0,trace_len*cv_factor,cv_factor)]
    bins = range(50)
    plt.plot(fluo_noise)
    plt.plot(fluo_raw)
    plt.show()