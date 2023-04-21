import pandas as pd
import numpy as np

def GetDfs(category="HF|LF|NW", response="1|0",
           accuracy="1|0", pred_df_type='rt'):

    experiment_df = behavioural_df.loc[(behavioural_df['category'].str.match(category)) &
                                       (behavioural_df['response'].apply(str).str.match(str(response))) &
                                       (behavioural_df['accuracy'].apply(str).str.match(str(accuracy)))]

    if response == "1|0":
        predicted_df = predictedData[pred_df_type][behavioural_df.loc[behavioural_df['category'].str.match(
            category)].index]
    else:
        predicted_df = predictedData[pred_df_type][predictedData['response'] == int(
            response)][behavioural_df.loc[behavioural_df['category'].str.match(category)].index]

    return (experiment_df, predicted_df)


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
