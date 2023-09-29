import os
import pandas as pd
import numpy as np
import cmdstanpy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

def get_rt_quantiles(behavioural_df, probs, result_df_size = 400, method="median_unbiased"):
    df = behavioural_df.copy()
    final_df = pd.DataFrame([])
    rt_array = df['rt'].to_numpy()
    quantiles = np.quantile(rt_array, probs, method=method)
    
    while len(final_df) < result_df_size and len(df) != 0:
        diff = result_df_size - len(final_df)
        distanceFromQuantiles = np.abs(rt_array[:, np.newaxis] - quantiles)
        indexes = distanceFromQuantiles.argmin(axis=0)
        if  diff < indexes.shape[0]:
            indexes = np.random.choice(indexes, diff)
        final_df = pd.concat([final_df, df.iloc[indexes, :]])
        df = df.drop(indexes).reset_index(drop=True)
        rt_array = df['rt'].to_numpy()
        
    return final_df

def remove_outliers(df, max_rt, min_rt, std_c=2.5):
    """
    Returns remove outliers from dataframes. Outlier RTs are bigger than
    max_rt and smaller than min_rt. Also RTsthat are out of -/+ (std_c * sd) 
    of mean RT interval are considered as outliers too.

    Parameters
    ----------
        df: pandas dataframe with rt column
        max_rt (float): maximum acceptable rt
        min_rt (float): minimum acceptable rt
        
    Optional Parameters
    ----------
        std_c (float) : Optional
            coefficient to define interval of non-outlier RTs
    
    Returns
    -------
        df: pandas dataframe without outliers  
    """
    mean = df['rt'].mean()
    sd = df['rt'].std()
    lower_thr = mean - std_c*sd
    upper_thr = mean + std_c*sd
    min_bound = max(min_rt, lower_thr)
    max_bound = min(max_rt, upper_thr)
    df = df[df['rt'] >= min_bound]
    df = df[df['rt'] <= max_bound]
    return df

def get_dfs(behavioural_df, predicted_data, category="HF|LF|NW", response="1|0",
           accuracy="1|0", pred_df_type='rt'):

    experiment_df = behavioural_df.loc[(behavioural_df['category'].str.match(category)) &
                                       (behavioural_df['response'].apply(str).str.match(str(response))) &
                                       (behavioural_df['accuracy'].apply(str).str.match(str(accuracy)))]

    if response == "1|0":
        predicted_df = predicted_data[pred_df_type][behavioural_df.loc[behavioural_df['category'].str.match(
            category)].index]
    else:
        predicted_df = predicted_data[pred_df_type][predicted_data['response'] == int(
            response)][behavioural_df.loc[behavioural_df['category'].str.match(category)].index]

    return (experiment_df, predicted_df)

def plot_mean_posterior(x, data_mean, ax):
    """
    Plots the posterior of x with experimental data mean as a line
    
    Parameters
    ----------
    x : array-like
        An array containing RT or response for each trial.
        
    x : float
        mean of RT or Accuracy of experimental data.

    ax : matplotlib.axes.Axes
        
    Returns
    -------
    None
    """
    density = gaussian_kde(x, bw_method='scott')
    xd = np.linspace(x.min(), x.max())
    yd = density(xd)

    low, high = bci(x)
    ax.fill_between(xd[np.logical_and(xd >= low, xd <= high)],
                     yd[np.logical_and(xd >= low, xd <= high)], color = 'lightsteelblue')

    ax.plot(xd, yd, color='slategray')
    ax.axvline(data_mean, color='red')

def bci(x, alpha=0.05):
    """
    Calculate Bayesian credible interval (BCI).

    Parameters
    ----------
    x : array-like
        An array containing MCMC samples.

    Optional Parameters
    -------------------
    alpha : float, default 0.05
        Desired probability of type I error.

    Returns
    -------
    interval : numpy.ndarray
        Array containing the lower and upper bounds of the bci interval.
    """
    interval = np.nanpercentile(x, [(alpha/2)*100, (1-alpha/2)*100])

    return interval

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width.
    Parameters
    ----------
    x : array-like
        An sorted numpy array.
    alpha : float
        Desired probability of type I error (defaults to 0.05).
    Returns
    -------
    hdi_min : float
        The lower bound of the interval.
    hdi_max : float
        The upper bound of the interval.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max

def hdi(x, alpha=0.05):
    """Calculate highest posterior density (HPD).
        Parameters
        ----------
        x : array-like
            An array containing MCMC samples.
        alpha : float
            Desired probability of type I error (defaults to 0.05).
    Returns
    -------
    interval : numpy.ndarray
        Array containing the lower and upper bounds of the hdi interval.
    """

    # Make a copy of trace
    x = x.copy()
     # Sort univariate node
    sx = np.sort(x)
    interval = np.array(calc_min_interval(sx, alpha))

    return interval

def calculate_waic(log_likelihood, pointwise=False):
    """
    Returns model comparisions' metrics.
    
    Parameters
    ----------
        log_likelihood: np.array
            log_likelihood of each trial
        max_rt: float
            maximum acceptable rt
        min_rt: float
             minimum acceptable rt
             
    Optional Parameters
    ----------------
    pointwise: float
        if true pointwise waic will be calculated
        
    Returns
    -------
        out:  a dictionary containing lppd, waic, waic_se and pointwise_waic    
    """
    likelihood = np.exp(log_likelihood)

    mean_l = np.mean(likelihood, axis=0) # N observations

    pointwise_lppd = np.log(mean_l)
    lppd = np.sum(pointwise_lppd)

    pointwise_var_l = np.var(log_likelihood, axis=0) # N observations
    var_l = np.sum(pointwise_var_l)

    pointwise_waic = - 2*pointwise_lppd +  2*pointwise_var_l
    waic = -2*lppd + 2*var_l
    waic_se = np.sqrt(log_likelihood.shape[1] * np.var(pointwise_waic))

    if pointwise:
        out = {'lppd':lppd,
               'p_waic':var_l,
               'waic':waic,
               'waic_se':waic_se,
               'pointwise_waic':pointwise_waic}
    else:
        out = {'lppd':lppd,
               'p_waic':var_l,
                'waic':waic,
                'waic_se':waic_se}
    return out

def get_parameters_range(path_to_stan_output, behavioural_df):
    fit = cmdstanpy.from_csv(path_to_stan_output)
    params = fit.stan_variables()
    particpant_trials_counts = np.unique(behavioural_df.participant_id,
                                         return_counts=True)[1]

    max_likelihood_participant = 0
    max_likelihood = -float("inf")
    lower_index = 0

    for participant_index, trials_count in enumerate(particpant_trials_counts):
        upper_index = lower_index + trials_count
        sum_likelihood = np.sum(params["log_lik"][:, lower_index:upper_index])
        if sum_likelihood > max_likelihood:
            max_likelihood_participant = participant_index
        lower_index = upper_index
    
    parameters_range_df = pd.DataFrame([], columns=["mean", "std"])
    needed_parameters = ["alpha_sbj", "b_sbj", "k_1_sbj", "k_2_sbj",
                        "threshold_sbj_word", "threshold_sbj_nonword",
                        "ndt_sbj"]

    for parameter in needed_parameters:
        max_val = params[parameter][:, max_likelihood_participant].mean()
        min_val = params[parameter][:, max_likelihood_participant].std()
        parameters_range_df.loc[parameter] = [max_val, min_val]

    return parameters_range_df

def get_stan_parameters(generated_df, priors):
    N = len(generated_df)                                                    # For all models
    p = generated_df.loc[:, ["word_prob", "non_word_prob"]].to_numpy()       # predicted probabilites of words and non-words, for ANN-EAM models
    frequency = generated_df["zipf"].to_numpy().astype(int)                  # zipf values, for models with non-decision time or drift modulation
    frequencyCondition = generated_df["category"].replace(["HF", "LF", "NW"], [1, 2, 3]).to_numpy() # For models with conditional drift
    response = generated_df["response"].to_numpy().astype(int)               # for all models
    rt = generated_df["rt"].to_numpy()                                       # for all models
    minRT = generated_df["minRT"].to_numpy()                                 # for all models
    RTbound = 0.05                                                           # for all models
    ndt = generated_df["ndt"]
    threshold_word = generated_df["threshold_word"]
    threshold_nonword = generated_df["threshold_nonword"]
    k_2 = generated_df["k_2"]
    b = generated_df["b"]
    alpha = generated_df["alpha"]
    k_1 = generated_df["k_1"]
    

    # define input for the model
    data_dict = {"N": N,
                 "response": response,
                 "rt": rt,
                 "minRT": minRT,
                 "RTbound": RTbound,
                 "frequency": frequency,
                 "frequencyCondition": frequencyCondition,
                 "p": p,
                 "ndt":ndt,
                 "threshold_word": threshold_word,
                 "threshold_nonword": threshold_nonword,
                 "k_2": k_2,
                 "b":b,
                 "alpha": alpha,
                 "k_1": k_1,
                 "threshold_priors": priors['threshold_priors'],
                 "ndt_priors":  priors['ndt_priors'],
                 "alpha_priors": priors['alpha_priors'],
                 "b_priors": priors['b_priors'],
                 "k_priors": priors['k_priors']
                 }
    return data_dict

def save_results_to_csv(fit, parameters_set, col):
    columns = {}
    
    recoverd_df = pd.DataFrame([], columns=columns.keys())
    stan_variables = fit.stan_variables()
    
    columns['mean_' + col] = stan_variables[col].mean()
    columns['median_' + col] = np.median(stan_variables[col])
    columns['real_' + col] = parameters_set.loc[col, "generated"]
    columns['mean_' + col] = stan_variables[col].mean()
    columns['HDI_' + col + '_bottom'], columns['HDI_'+ col + '_top'] = hdi(stan_variables[col])
    
    
    recoverd_df = pd.concat([recoverd_df, pd.DataFrame(columns, index=[0])])

    output_path='RecoveryResults/' + col + '_recovery_results.csv'
    recoverd_df.to_csv(output_path,
                       mode="a",
                       header=not os.path.exists(output_path),
                       index=False)

def plot_parameter_recovery_results(param_name):
    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["pdf.use14corefonts"] = True


    recovery_data = pd.read_csv('RecoveryResults/' + param_name + '_recovery_results.csv', header=0)
    
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    posterior_mean = recovery_data['mean_' + param_name]
    posterior_median = recovery_data['median_' + param_name]
    true = recovery_data['real_' + param_name]
    ax.scatter(true, posterior_mean, color="tomato",
                                    zorder=10)
    ax.scatter(true, posterior_median, color="black",
                                    zorder=9)
    ax.vlines(x=true.to_numpy(), linewidth=2,
              ymin=recovery_data['HDI_' + param_name + '_bottom'].to_numpy(),
              ymax=recovery_data['HDI_'+ param_name + '_top'].to_numpy())
    ax.set_title(param_name)

    min_true_point = true.min()
    max_true_point = true.max()
    recoverline = ax.axline((min_true_point, min_true_point),
                            (max_true_point, max_true_point))
    plt.setp(recoverline, linewidth=3, color="grey")

    r2 = r2_score(true, posterior_mean)
    ax.text(0.08, 0.92, f"R2: {r2:.2f}", horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)

    custom_xlim = (min(min_true_point, posterior_mean.min())-0.5,
                   max(max_true_point, posterior_mean.max())+0.5)
    custom_ylim = (min(min_true_point, posterior_mean.min())-0.5,
                   max(max_true_point, posterior_mean.max())+0.5)

    # Setting the values for all axes.
    plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
    plt.savefig('Plots/' + param_name + '_recovery.pdf')

