import numpy as np
import pandas as pd

def random_rdm_2A(w_drift, nw_drift, threshold_word, threshold_nonword, ndt, noise_constant=1, dt=0.001, max_rt=10):
    """ 
    Simulates behavior (rt and accuracy) according to the Racing Diffusion Model.

    Parameters
    ----------
    cor_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - correct trials.
    inc_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - incorrect trials.
    threshold : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Threshold of the diffusion decision model.
    ndt : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Non decision time of the diffusion decision model, in seconds.

    Optional Parameters
    ----------------
    noise_constant : float, default 1
        Scaling factor of the Racing Diffusion Model.
        If changed, drift and threshold would be scaled accordingly.
        Not to be changed in most applications.
    dt : float, default 0.001
        Controls the time resolution of the Racing Diffusion Model. Default is 1 msec.
        Lower values of dt make the function more precise but much slower.
    max_rt : float, default 10
        Controls the maximum rts that can be predicted.
        Making this higher might make the function a bit slower.

    Returns
    -------
    rt : numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response times according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.
    acc: numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated accuracy according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.
    """
    shape = w_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    max_tsteps = max_rt/dt

    x_cor = np.zeros(shape)
    x_inc = np.zeros(shape)

    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x_cor[ongoing] += np.random.normal(w_drift[ongoing]*dt,
                                           noise_constant*np.sqrt(dt),
                                           np.sum(ongoing))
        x_inc[ongoing] += np.random.normal(nw_drift[ongoing]*dt,
                                           noise_constant*np.sqrt(dt),
                                           np.sum(ongoing))
        tstep += 1
        ended_correct = (x_cor >= threshold_word)
        ended_incorrect = (x_inc >= threshold_nonword)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt * \
                tstep + ndt[np.logical_and(ended_correct, ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt * \
                tstep + ndt[np.logical_and(ended_incorrect, ongoing)]
            ongoing[ended_incorrect] = False
    return rt, acc

def simulate_ANNRDM_individual(n_trials, trials_info_df, parameters_set):
    
    data = pd.DataFrame([])    
    selected_trials_df = trials_info_df.sample(n_trials).reset_index(drop=True)
    zipf = selected_trials_df["zipf"].to_numpy()
    pword = selected_trials_df["word_prob"].to_numpy()
    pnword = selected_trials_df["non_word_prob"].to_numpy()     
    
    data["trial"] = np.arange(1, n_trials+1)

    alpha = parameters_set.loc["alpha", "generated"]
    b = parameters_set.loc["b", "generated"]
    k_1 = parameters_set.loc["k_1", "generated"]
    k_2 = parameters_set.loc["k_2", "generated"]
    threshold_word = parameters_set.loc["threshold_word", "generated"]
    threshold_nonword = parameters_set.loc["threshold_nonword", "generated"]
    g = parameters_set.loc["g", "generated"]
    m = parameters_set.loc["m", "generated"]
    
    data["k_1"] = np.repeat(k_1, n_trials)
    data["k_2"] = np.repeat(k_2, n_trials)
    data["alpha"]= np.repeat(alpha, n_trials)
    data["b"] = np.repeat(b, n_trials)
    data["m"] = np.repeat(m, n_trials)
    data["g"] = np.repeat(g, n_trials)
    data["threshold_word"] = np.repeat(threshold_word, n_trials)
    data["threshold_nonword"] = np.repeat(threshold_nonword, n_trials)

    word_drifts = k_1 + b * zipf
    word_drifts /= 1 + np.exp(-alpha * (pword-0.5))
    data["word_drifts"] = word_drifts

    nonword_drifts = k_2 / (1 + np.exp(-alpha * (pnword-0.5)))
    data["nonword_drifts"] = nonword_drifts

    data["ndt"] = m + g * np.exp(-zipf)
    
    rt, response = random_rdm_2A(data["word_drifts"], data["nonword_drifts"],
                      data["threshold_word"], data["threshold_nonword"],
                      data["ndt"], max_rt=10)
    
    selected_trials_df_edited =  selected_trials_df[["zipf", "category", "word_prob", "non_word_prob"]].reset_index().drop(["index"], axis=1)
    data = pd.concat([data, selected_trials_df_edited], axis=1)
    
    data["rt"] = rt
    data["response"] = response
    min_rt = data["rt"].min()
    data["minRT"] = np.repeat(min_rt, n_trials)
    
    return data


def simulate_ANNRDM(n_trials, trials_info_df, parameters_set):
    
    data = pd.DataFrame([])    
    n_participants = parameters_set.shape[0]

    selected_trials_df = trials_info_df.sample(n_trials).reset_index(drop=True)
    zipf = selected_trials_df["zipf"].to_numpy()
    pword = selected_trials_df["word_prob"].to_numpy()
    pnword = selected_trials_df["non_word_prob"].to_numpy()     
    
    data["participant_id"] = np.repeat(np.arange(n_participants)+1, n_trials)
    data["trial"] = np.tile(np.arange(1, n_trials+1), n_participants)

    alpha_sbj = parameters_set[:,0]
    b_sbj = parameters_set[:, 1]
    k_1_sbj = parameters_set[:, 2]
    k_2_sbj = parameters_set[:, 3]
    threshold_word_sbj = parameters_set[:, 4]
    threshold_nonword_sbj = parameters_set[:, 5]
    m_sbj = parameters_set[:, 6]
    g_sbj = parameters_set[:, 7]
    
    data["k_1"] = np.repeat(k_1_sbj, n_trials)
    data["k_2"] = np.repeat(k_2_sbj, n_trials)
    data["alpha"]= np.repeat(alpha_sbj, n_trials)
    data["b"] = np.repeat(b_sbj, n_trials)
    data["m"] = np.repeat(m_sbj, n_trials)
    data["g"] = np.repeat(g_sbj, n_trials)
    data["threshold_word"] = np.repeat(threshold_word_sbj, n_trials)
    data["threshold_nonword"] = np.repeat(threshold_nonword_sbj, n_trials)

    word_drifts = k_1_sbj[:, np.newaxis] + b_sbj[:,np.newaxis] * zipf
    word_drifts /= 1 + np.exp((pword-0.5) * -alpha_sbj[:, np.newaxis])
    word_drifts = word_drifts.ravel()
    data["word_drifts"] = word_drifts

    nonword_drifts = k_2_sbj[:, np.newaxis] / (1 + np.exp((pnword-0.5) * -alpha_sbj[:, np.newaxis]))
    nonword_drifts = nonword_drifts.ravel()
    data["nonword_drifts"] = nonword_drifts

    data["ndt"] = (m_sbj[:, np.newaxis] + g_sbj[:, np.newaxis] * np.exp(-zipf)).ravel()
    
    rt, response = random_rdm_2A(data["word_drifts"], data["nonword_drifts"],
                      data["threshold_word"], data["threshold_nonword"],
                      data["ndt"], max_rt=10)
    
    selected_trials_df_edited =  pd.concat([selected_trials_df[["zipf", "category", "word_prob", "non_word_prob"]]]
                                 *n_participants).reset_index().drop(["index"], axis=1)
    data = pd.concat([data, selected_trials_df_edited], axis=1)
    
    data["rt"] = rt
    data["response"] = response
    min_rt = data.groupby("participant_id")["rt"].min().to_numpy()
    data["minRT"] = np.repeat(min_rt, n_trials)
    
    return data

def random_lba_2A(word_drift, nonword_drift, sp_trial_var_word, sp_trial_var_nonword,
                  ndt, k_word, k_nonword, drift_trial_var):
    """Simulates behavior (rt and accuracy) according to the Linear Ballistic Accumulator.
    Parameters
    ----------
    word_drift : numpy.ndarray
        Drift-rate of the Linear Ballistic Accumulator - correct responses. 1D array of floats.
    nonword_drift : numpy.ndarray
        Drift-rate of the Linear Ballistic Accumulator - incorrect responses. 1D array of floats.
    sp_trial_var_word : float
        Starting point variability of the Linear Ballistic Accumulator for words. Also called A.
    sp_trial_var_nonword : float
        Starting point variability of the Linear Ballistic Accumulator for non-words. Also called A.    
    ndt : float
        Non-decision time of the Linear Ballistic Accumulator. Also called tau.
    k_word : float
        Distance between starting point variability and threshold for words.
    k_nonword : float
        Distance between starting point variability and threshold for non-words.
    drift_trial_var : numpy.ndarray, default None
        The drift rate trial variability. 1D array of 0s and 1s.
    Returns
    -------
    rt : numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response times according to the Linear Ballistic Accumulator.
        Every element corresponds to the set of parameters given as input with the same shape.
    resp: numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response according to the Linear Ballistic Accumulator.
        Every element corresponds to the set of parameters given as input with the same shape.
    """
    shape = word_drift.shape
    resp = np.empty(shape)
    rt = np.empty(shape)
    resp[:] = np.nan
    rt[:] = np.nan

    b_word = k_word + sp_trial_var_word
    b_nonword = k_nonword + sp_trial_var_nonword
    one_pose = True
    v_word = np.array(word_drift)
    v_nonword = np.array(nonword_drift)

    # this while loop might be wrong
    while one_pose:
        ind = np.logical_and(v_word < 0, v_nonword < 0)
        if drift_trial_var is None:
            v_word[ind] = np.random.normal(
                word_drift[ind], np.ones(word_drift[ind].shape))
            v_nonword[ind] = np.random.normal(
                nonword_drift[ind], np.ones(nonword_drift[ind].shape))
        else:
            v_word[ind] = np.random.normal(
                word_drift[ind], drift_trial_var[ind])
            v_nonword[ind] = np.random.normal(
                nonword_drift[ind], drift_trial_var[ind])

        one_pose = np.sum(ind) > 0

    start_word = np.random.uniform(
        np.zeros(sp_trial_var_word.shape), sp_trial_var_word)
    start_non_word = np.random.uniform(
        np.zeros(sp_trial_var_nonword.shape), sp_trial_var_nonword)

    ttf_word = (b_word - start_word) / v_word
    ttf_nonword = (b_nonword - start_non_word) / v_nonword

    ind = np.logical_and(ttf_word <= ttf_nonword, 0 < ttf_word)
    resp[ind] = 1
    rt[ind] = ttf_word[ind] + ndt[ind]

    ind = np.logical_and(ttf_nonword < 0, 0 < ttf_word)
    resp[ind] = 1
    rt[ind] = ttf_word[ind] + ndt[ind]

    ind = np.logical_and(ttf_nonword < ttf_word, 0 < ttf_nonword)
    resp[ind] = 0
    rt[ind] = ttf_nonword[ind] + ndt[ind]

    ind = np.logical_and(ttf_word < 0, 0 < ttf_nonword)
    resp[ind] = 0
    rt[ind] = ttf_nonword[ind] + ndt[ind]

    return rt, resp

    pass