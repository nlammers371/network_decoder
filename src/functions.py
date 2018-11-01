import numpy as np

def gillespie_trace_discrete_loading(memory, trace_len, Tres, R, r ,noise , alpha=0.0, AUperPolII = 20.0):
    """
    This script simulates stochastic transcriptional activity with discrete Pol II loading events
    For now loading events are assumed to follow Poisson statistics
    :param memory: Pol II elongation time (time steps)
    :param trace_len: Trace length (in time steps)
    :param Tres: Sampling resolution (in seconds)
    :param R: Transition rate matrix (1/s)
    :param r: Initiation rate vector (PolII/s)
    :param noise: Additive gaussian noise
    :param alpha: MS2 rise time (time steps)
    :param AUperPolII: Calibration factor
    :return:
    """
    # generate convultion kernel to deal with MS2 rise time
    if alpha > 0:
        alpha_vec = [(float(i + 1) / alpha + (float(i) / alpha)) / 2.0 * (i < alpha) * ((i + 1) <= alpha)
                     + ((alpha - i)*(1 + float(i) / alpha) / 2.0 + i + 1 - alpha) * (i < alpha) * (i + 1 > alpha)
                     + 1 * (i >= alpha) for i in range(memory)]

        #alpha_vec = np.array(alpha_vec[::-1])
    else:
        alpha_vec = np.array([1.0]*memory)

    kernel = np.ones(memory)*alpha_vec
    kernel_unif = np.ones(memory)

    # initialize lists and useful params
    K = int(len(r))
    trajectory = np.zeros((1,trace_len), dtype='int')
    promoter_states = []

    # Generate promoter trajectory
    T_float = 0.0
    transitions = [0.0] # track switch times
    loading_times = [];
    p_curr = np.random.choice(K)
    promoter_states.append(r[p_curr])
    while T_float < trace_len*Tres:
        # time step
        rdm = np.random.random()
        tau = 1 / -R[p_curr, p_curr]
        t = tau * math.log(1.0 / rdm) # determine wait time
        transitions.append(T_float + t)
        # simulate initiation events
        t_load = 0
        if r[p_curr] > 0:
            while t_load <= t:
                rdm = np.random.random()
                l_tau = 1 / r[p_curr]
                l_t = l_tau * math.log(1.0 / rdm)
                t_load += l_t
                if t_load <= t:
                    loading_times.append(T_float+t_load)
        # simulate next transition
        p_probs = R[:, p_curr] / -R[p_curr, p_curr]
        p_probs[p_curr] = 0
        p_curr = np.random.choice(K, p=p_probs) # select state
        promoter_states.append(r[p_curr]*AUperPolII)


        T_float += t

    bins = np.linspace(0,trace_len*Tres,trace_len+1)
    promoter_grid_ld, bin_edges = np.histogram(loading_times, bins=bins)
    tr_array = np.array(transitions)
    promoter_states = promoter_states[:-1]
    promoter_grid = np.zeros(trace_len)

    # find total production for each time step
    for e in range(1, trace_len):
        # Find transitions that occurred within preceding time step
        if e == 1:
            tr_prev = 0
        else:
            # find most recent preceding switch
            tr_prev = np.max(np.where(tr_array < e*Tres - 1)[0])

        tr_post = np.min(np.where(tr_array >= e*Tres)[0])
        # get total fluorescence for each step
        tr = transitions[tr_prev:tr_post + 1]
        tr[0] = (e - 1)*Tres
        tr[-1] = e*Tres
        tr_diffs = np.diff(tr)
        p_states = promoter_states[tr_prev:tr_post]
        promoter_grid[e] = np.sum(tr_diffs * p_states)

    promoter_grid.astype('int')
    # convolve to get fluo
    fluo_raw = np.convolve(kernel, promoter_grid, mode='full')
    fluo_raw = fluo_raw[0:trace_len]
    fluo_unif = np.convolve(kernel_unif, promoter_grid, mode='full')
    fluo_unif = fluo_unif[0:trace_len]
    fluo_raw_discrete = np.convolve(kernel, promoter_grid_ld, mode='full')
    fluo_raw_discrete = fluo_raw_discrete[0:trace_len]*AUperPolII
    # add noise
    noise_vec = np.random.randn(trace_len) * noise
    fluo_noise = fluo_raw + noise_vec
    fluo_noise_discrete = fluo_raw_discrete + noise_vec
    # output
    return fluo_noise, fluo_noise_discrete, fluo_raw, fluo_raw_discrete, \
            fluo_unif, promoter_grid, promoter_grid_ld, transitions, promoter_states, loading_times
