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
from scipy.stats import multinomial
from sklearn.preprocessing import normalize

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


def simulate_initiation_states(trace_len, A, p_vec):
    """
    :param trace_len: number of time steps in trace
    :param A: transition probability matrix (KxK)
    :param p_vec: initiation probability vector (Kx1)
    :return: system_states: 1xT integer vector (K possible values) indicating kinetic state of system at each time step
             initiation_states: 1xT  binary vector indicating presence or absence of initiation event
    """

    # initialize lists and useful params
    K = A.shape[0]
    # track promoter states
    system_states = np.empty(trace_len, dtype='int')
    # track promoter initiation states
    initiation_states = np.empty(trace_len)
    # draw first system state
    system_states[0] = np.random.choice(K)
    # draw initiation state conditional on system state
    initiation_states[0] = np.random.choice([0, 1], 1, p=[1-p_vec[system_states[0]], p_vec[system_states[0]]])

    for i in range(1, trace_len):
        # time step
        system_states[i] = np.random.choice(K, 1, p=A[:, system_states[i-1]])
        initiation_states[i] = np.random.choice([0, 1], 1, p=[1-p_vec[system_states[i]], p_vec[system_states[i]]])

    # output
    return system_states, initiation_states
