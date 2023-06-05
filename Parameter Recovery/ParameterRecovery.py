import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

import numpy as np
import pandas as pd
import cmdstanpy 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import os
import json
from utils.random import simulate_ANNRDM_individual
from utils.utils import get_parameters_range, hdi
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["pdf.use14corefonts"] = True

root = "../"
plots_root = "Results/individual/Plots/"
datasets_root = root + "Datasets/"
behavioural_data_root = datasets_root +  "behavioral_data/selected_data/" 

dataset_path = datasets_root + "AI Models Results/fastText_FC.csv"


word_nword_df = pd.read_csv(dataset_path, header=None,
                            names =["string", "freq",  "label", "zipf",
                                    "category", "word_prob", "non_word_prob"])

number_of_participants = 100
n_trials = 400

# set sampling parameters
n_iter = 7000
n_warmup = int(n_iter/2)
n_sample = int(n_iter/2)
n_chains = 4

threshold_priors = [2, 1]          # For all models with RDM
ndt_priors = [0, 1];               # For models wtihout non-decision time modulation
g_priors = [-2, 1]                 # For models wtih non-decision time modulation
m_priors = [0, 0.5]                # For models wtih non-decision time modulation
drift_priors = [1, 2]              # For models without drift mapping functions (non ANN-EAM models)
alpha_priors = [0, 1]              # For models with drift mapping functions
b_priors = [0, 1]                  # For models with drift mapping functions with asymptote modulation and linear models
k_priors = [2, 1]                  # For models with sigmoid drift mapping functions (ANN-EAM models)

def get_stan_parameters(generated_df):
    N = len(generated_df)                                                    # For all models
    p = generated_df.loc[:, ["word_prob", "non_word_prob"]].to_numpy()       # predicted probabilites of words and non-words, for ANN-EAM models
    frequency = generated_df["zipf"].to_numpy().astype(int)                  # zipf values, for models with non-decision time or drift modulation
    frequencyCondition = generated_df["category"].replace(["HF", "LF", "NW"], [1, 2, 3]).to_numpy() # For models with conditional drift
    response = generated_df["response"].to_numpy().astype(int)               # for all models
    rt = generated_df["rt"].to_numpy()                                       # for all models
    minRT = generated_df["minRT"].to_numpy()                                 # for all models
    RTbound = 0.1                                                             # for all models
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
                 "threshold_priors": threshold_priors,
                 "ndt_priors": ndt_priors,
                 "g_priors": g_priors,
                 "m_priors": m_priors,
                 "drift_priors": drift_priors,
                 "p": p,
                 "alpha_priors": alpha_priors,
                 "b_priors": b_priors,
                 "k_priors": k_priors,
                 "ndt":ndt,
                 "threshold_word": threshold_word,
                 "threshold_nonword": threshold_nonword,
                 "k_2": k_2,
                 "b":b,
                 "alpha": alpha,
                 "k_1": k_1
                 }
    return data_dict

# K_1
stan_file_path = root +  "models/stan/ANN-RDM/individual/k_1_recovery.stan" 

def save_results_to_csv(fit, parameters_set):
    columns = {}
    
    recoverd_df = pd.DataFrame([], columns=columns.keys())
    stan_variables = fit.stan_variables()

    columns["mean_k_1"] = stan_variables["k_1"].mean()
    columns["median_k_1"] = np.median(stan_variables["k_1"])
    columns["real_k_1"] = parameters_set.loc["k_1", "generated"]
    columns["HDI_k_1_bottom"], columns["HDI_k_1_top"] = hdi(stan_variables["k_1"])

    
    recoverd_df = pd.concat([recoverd_df, pd.DataFrame(columns, index=[0])])

    output_path="RecoveryResults/k_1_recovery_results.csv"
    recoverd_df.to_csv(output_path,
                       mode="a",
                       header=not os.path.exists(output_path),
                       index=False)
    
## Simulation and Estimation process
rdm_model = cmdstanpy.CmdStanModel(model_name="ANN-RDM_full_FC",
                                   stan_file=stan_file_path)

iteration_count = 0
params_range = pd.read_csv("Data/params_range.csv", index_col=0)
while iteration_count < number_of_participants:
    print(f"Iteration for participant {iteration_count+1} Started")
    parameters_set = params_range.copy()
    parameters_set["generated"] = np.random.normal(loc=parameters_set.iloc[:, 0],
                                                   scale=parameters_set.iloc[:, 1])

    behavioral_df = simulate_ANNRDM_individual(n_trials=n_trials, trials_info_df=word_nword_df,
                                         parameters_set=parameters_set)
    stan_parameters = get_stan_parameters(behavioral_df)
    fit = rdm_model.sample(data=stan_parameters,
                       iter_sampling=n_sample, 
                       iter_warmup=n_warmup,
                       chains=n_chains,
                       show_console=False)
    
    df = fit.summary()
    badRhat = False
    for f in df["R_hat"]:
        if f >= 1.01 or f <= 0.9:
            badRhat = True
    
    if badRhat:
        print("Split R-hat values are not satisfactory for all parameters. repeating iteration") 
    else:
        save_results_to_csv(fit, parameters_set)
        print(f"Iteration for participant {iteration_count+1} Finished") 
        iteration_count += 1

## Particpants parameter recovery
recovery_data = pd.read_csv("RecoveryResults/k_1_recovery_results.csv", header=0)

fig, ax = plt.subplots(1, 1, figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
posterior_mean = recovery_data["mean_k_1"]
posterior_median = recovery_data["median_k_1"]
true = recovery_data["real_k_1"]
ax.scatter(true, posterior_mean, color="tomato",
                                zorder=10)
ax.scatter(true, posterior_median, color="yellow",
                                zorder=9)
ax.vlines(x=true.to_numpy(), linewidth=2,
          ymin=recovery_data["HDI_k_1_bottom"].to_numpy(),
          ymax=recovery_data["HDI_k_1_top"].to_numpy())
ax.set_title("k_1")

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
plt.savefig("Plots/k_1_recovery.pdf")


# K_2
stan_file_path = root +  "models/stan/ANN-RDM/individual/k_2_recovery.stan" 

def save_results_to_csv(fit, parameters_set):
    columns = {}
    
    recoverd_df = pd.DataFrame([], columns=columns.keys())
    stan_variables = fit.stan_variables()

    columns["mean_k_2"] = stan_variables["k_2"].mean()
    columns["median_k_2"] = np.median(stan_variables["k_2"])
    columns["real_k_2"] = parameters_set.loc["k_2", "generated"]
    columns["HDI_k_2_bottom"], columns["HDI_k_2_top"] = hdi(stan_variables["k_2"])

    
    recoverd_df = pd.concat([recoverd_df, pd.DataFrame(columns, index=[0])])

    output_path="RecoveryResults/k_2_recovery_results.csv"
    recoverd_df.to_csv(output_path,
                       mode="a",
                       header=not os.path.exists(output_path),
                       index=False)
    
## Simulation and Estimation process
rdm_model = cmdstanpy.CmdStanModel(model_name="ANN-RDM_full_FC",
                                   stan_file=stan_file_path)

iteration_count = 0
params_range = pd.read_csv("Data/params_range.csv", index_col=0)
while iteration_count < number_of_participants:
    print(f"Iteration for participant {iteration_count+1} Started")
    parameters_set = params_range.copy()
    parameters_set["generated"] = np.random.normal(loc=parameters_set.iloc[:, 0],
                                                   scale=parameters_set.iloc[:, 1])

    behavioral_df = simulate_ANNRDM_individual(n_trials=n_trials, trials_info_df=word_nword_df,
                                         parameters_set=parameters_set)
    stan_parameters = get_stan_parameters(behavioral_df)
    fit = rdm_model.sample(data=stan_parameters,
                       iter_sampling=n_sample, 
                       iter_warmup=n_warmup,
                       chains=n_chains,
                       show_console=False)
    
    df = fit.summary()
    badRhat = False
    for f in df["R_hat"]:
        if f >= 1.01 or f <= 0.9:
            badRhat = True
    
    if badRhat:
        print("Split R-hat values are not satisfactory for all parameters. repeating iteration") 
    else:
        save_results_to_csv(fit, parameters_set)
        print(f"Iteration for participant {iteration_count+1} Finished") 
        iteration_count += 1
        
## Particpants parameter recovery
recovery_data = pd.read_csv("RecoveryResults/k_2_recovery_results.csv", header=0)

fig, ax = plt.subplots(1, 1, figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
posterior_mean = recovery_data["mean_k_2"]
posterior_median = recovery_data["median_k_2"]
true = recovery_data["real_k_2"]
ax.scatter(true, posterior_mean, color="tomato",
                                zorder=10)
ax.scatter(true, posterior_median, color="yellow",
                                zorder=9)
ax.vlines(x=true.to_numpy(), linewidth=2,
          ymin=recovery_data["HDI_k_2_bottom"].to_numpy(),
          ymax=recovery_data["HDI_k_2_top"].to_numpy())
ax.set_title("k_2")

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
plt.savefig("Plots/k_2_recovery.pdf")


# b
stan_file_path = root +  "models/stan/ANN-RDM/individual/b_recovery.stan"

def save_results_to_csv(fit, parameters_set):
    columns = {}
    
    recoverd_df = pd.DataFrame([], columns=columns.keys())
    stan_variables = fit.stan_variables()

    columns["mean_b"] = stan_variables["b"].mean()
    columns["median_b"] = np.median(stan_variables["b"])
    columns["real_b"] = parameters_set.loc["b", "generated"]
    columns["HDI_b_bottom"], columns["HDI_b_top"] = hdi(stan_variables["b"])

    
    recoverd_df = pd.concat([recoverd_df, pd.DataFrame(columns, index=[0])])

    output_path="RecoveryResults/b_recovery_results.csv"
    recoverd_df.to_csv(output_path,
                       mode="a",
                       header=not os.path.exists(output_path),
                       index=False)
    
## Simulation and Estimation process
rdm_model = cmdstanpy.CmdStanModel(model_name="ANN-RDM_full_FC",
                                   stan_file=stan_file_path)

iteration_count = 0
params_range = pd.read_csv("Data/params_range.csv", index_col=0)
while iteration_count < number_of_participants:
    print(f"Iteration for participant {iteration_count+1} Started")
    parameters_set = params_range.copy()
    parameters_set["generated"] = np.random.normal(loc=parameters_set.iloc[:, 0],
                                                   scale=parameters_set.iloc[:, 1])

    behavioral_df = simulate_ANNRDM_individual(n_trials=n_trials, trials_info_df=word_nword_df,
                                         parameters_set=parameters_set)
    stan_parameters = get_stan_parameters(behavioral_df)
    fit = rdm_model.sample(data=stan_parameters,
                       iter_sampling=n_sample, 
                       iter_warmup=n_warmup,
                       chains=n_chains,
                       show_console=False)
    
    df = fit.summary()
    badRhat = False
    for f in df["R_hat"]:
        if f >= 1.01 or f <= 0.9:
            badRhat = True
    
    if badRhat:
        print("Split R-hat values are not satisfactory for all parameters. repeating iteration") 
    else:
        save_results_to_csv(fit, parameters_set)
        print(f"Iteration for participant {iteration_count+1} Finished") 
        iteration_count += 1
        
## Particpants parameter recovery
recovery_data = pd.read_csv("RecoveryResults/b_recovery_results.csv", header=0)

fig, ax = plt.subplots(1, 1, figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
posterior_mean = recovery_data["mean_b"]
posterior_median = recovery_data["median_b"]
true = recovery_data["real_b"]
ax.scatter(true, posterior_mean, color="tomato",
                                zorder=10)
ax.scatter(true, posterior_median, color="yellow",
                                zorder=9)
ax.vlines(x=true.to_numpy(), linewidth=2,
          ymin=recovery_data["HDI_b_bottom"].to_numpy(),
          ymax=recovery_data["HDI_b_top"].to_numpy())
ax.set_title("b")

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
plt.savefig("Plots/b_recovery.pdf")


# alpha
stan_file_path = root +  "models/stan/ANN-RDM/individual/alpha_recovery.stan" 

def save_results_to_csv(fit, parameters_set):
    columns = {}
    
    recoverd_df = pd.DataFrame([], columns=columns.keys())
    stan_variables = fit.stan_variables()

    columns["mean_alpha"] = stan_variables["alpha"].mean()
    columns["median_alpha"] = np.median(stan_variables["alpha"])
    columns["real_alpha"] = parameters_set.loc["alpha", "generated"]
    columns["HDI_alpha_bottom"], columns["HDI_alpha_top"] = hdi(stan_variables["alpha"])

    
    recoverd_df = pd.concat([recoverd_df, pd.DataFrame(columns, index=[0])])

    output_path="RecoveryResults/alpha_recovery_results.csv"
    recoverd_df.to_csv(output_path,
                       mode="a",
                       header=not os.path.exists(output_path),
                       index=False)
    
## Simulation and Estimation process
rdm_model = cmdstanpy.CmdStanModel(model_name="ANN-RDM_full_FC",
                                   stan_file=stan_file_path)

iteration_count = 0
params_range = pd.read_csv("Data/params_range.csv", index_col=0)
while iteration_count < number_of_participants:
    print(f"Iteration for participant {iteration_count+1} Started")
    parameters_set = params_range.copy()
    parameters_set["generated"] = np.random.normal(loc=parameters_set.iloc[:, 0],
                                                   scale=parameters_set.iloc[:, 1])

    behavioral_df = simulate_ANNRDM_individual(n_trials=n_trials, trials_info_df=word_nword_df,
                                         parameters_set=parameters_set)
    stan_parameters = get_stan_parameters(behavioral_df)
    fit = rdm_model.sample(data=stan_parameters,
                       iter_sampling=n_sample, 
                       iter_warmup=n_warmup,
                       chains=n_chains,
                       show_console=False)
    
    df = fit.summary()
    badRhat = False
    for f in df["R_hat"]:
        if f >= 1.01 or f <= 0.9:
            badRhat = True
    
    if badRhat:
        print("Split R-hat values are not satisfactory for all parameters. repeating iteration") 
    else:
        save_results_to_csv(fit, parameters_set)
        print(f"Iteration for participant {iteration_count+1} Finished") 
        iteration_count += 1
        
## Particpants parameter recovery
recovery_data = pd.read_csv("RecoveryResults/alpha_recovery_results.csv", header=0)

fig, ax = plt.subplots(1, 1, figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
posterior_mean = recovery_data["mean_alpha"]
posterior_median = recovery_data["median_alpha"]
true = recovery_data["real_alpha"]
ax.scatter(true, posterior_mean, color="tomato",
                                zorder=10)
ax.scatter(true, posterior_median, color="yellow",
                                zorder=9)
ax.vlines(x=true.to_numpy(), linewidth=2,
          ymin=recovery_data["HDI_alpha_bottom"].to_numpy(),
          ymax=recovery_data["HDI_alpha_top"].to_numpy())
ax.set_title("alpha")

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
plt.savefig("Plots/alpha_recovery.pdf")
                                   
                            
# threshold_word
stan_file_path = root +  "models/stan/ANN-RDM/individual/threshold_word_recovery.stan" 

def save_results_to_csv(fit, parameters_set):
    columns = {}
    
    recoverd_df = pd.DataFrame([], columns=columns.keys())
    stan_variables = fit.stan_variables()

    columns["mean_threshold_word"] = stan_variables["threshold_word"].mean()
    columns["median_threshold_word"] = np.median(stan_variables["threshold_word"])
    columns["real_threshold_word"] = parameters_set.loc["threshold_word", "generated"]
    columns["HDI_threshold_word_bottom"], columns["HDI_threshold_word_top"] = hdi(stan_variables["threshold_word"])

    
    recoverd_df = pd.concat([recoverd_df, pd.DataFrame(columns, index=[0])])

    output_path="RecoveryResults/threshold_word_recovery_results.csv"
    recoverd_df.to_csv(output_path,
                       mode="a",
                       header=not os.path.exists(output_path),
                       index=False)

## Simulation and Estimation process
rdm_model = cmdstanpy.CmdStanModel(model_name="ANN-RDM_full_FC",
                                   stan_file=stan_file_path)

iteration_count = 0
params_range = pd.read_csv("Data/params_range.csv", index_col=0)
while iteration_count < number_of_participants:
    print(f"Iteration for participant {iteration_count+1} Started")
    parameters_set = params_range.copy()
    parameters_set["generated"] = np.random.normal(loc=parameters_set.iloc[:, 0],
                                                   scale=parameters_set.iloc[:, 1])

    behavioral_df = simulate_ANNRDM_individual(n_trials=n_trials, trials_info_df=word_nword_df,
                                         parameters_set=parameters_set)
    stan_parameters = get_stan_parameters(behavioral_df)
    fit = rdm_model.sample(data=stan_parameters,
                       iter_sampling=n_sample, 
                       iter_warmup=n_warmup,
                       chains=n_chains,
                       show_console=False)
    
    df = fit.summary()
    badRhat = False
    for f in df["R_hat"]:
        if f >= 1.01 or f <= 0.9:
            badRhat = True
    
    if badRhat:
        print("Split R-hat values are not satisfactory for all parameters. repeating iteration") 
    else:
        save_results_to_csv(fit, parameters_set)
        print(f"Iteration for participant {iteration_count+1} Finished") 
        iteration_count += 1
        
## Particpants parameter recovery
recovery_data = pd.read_csv("RecoveryResults/threshold_word_recovery_results.csv", header=0)

fig, ax = plt.subplots(1, 1, figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
posterior_mean = recovery_data["mean_threshold_word"]
posterior_median = recovery_data["median_threshold_word"]
true = recovery_data["real_threshold_word"]
ax.scatter(true, posterior_mean, color="tomato",
                                zorder=10)
ax.scatter(true, posterior_median, color="yellow",
                                zorder=9)
ax.vlines(x=true.to_numpy(), linewidth=2,
          ymin=recovery_data["HDI_threshold_word_bottom"].to_numpy(),
          ymax=recovery_data["HDI_threshold_word_top"].to_numpy())
ax.set_title("threshold_word")

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
plt.savefig("Plots/threshold_word_recovery.pdf")

# threshold_nonword
stan_file_path = root +  "models/stan/ANN-RDM/individual/threshold_nonword_recovery.stan" 

def save_results_to_csv(fit, parameters_set):
    columns = {}
    
    recoverd_df = pd.DataFrame([], columns=columns.keys())
    stan_variables = fit.stan_variables()

    columns["mean_threshold_nonword"] = stan_variables["threshold_nonword"].mean()
    columns["median_threshold_nonword"] = np.median(stan_variables["threshold_nonword"])
    columns["real_threshold_nonword"] = parameters_set.loc["threshold_nonword", "generated"]
    columns["HDI_threshold_nonword_bottom"], columns["HDI_threshold_nonword_top"] = hdi(stan_variables["threshold_nonword"])

    
    recoverd_df = pd.concat([recoverd_df, pd.DataFrame(columns, index=[0])])

    output_path="RecoveryResults/threshold_nonword_recovery_results.csv"
    recoverd_df.to_csv(output_path,
                       mode="a",
                       header=not os.path.exists(output_path),
                       index=False)

## Simulation and Estimation process
rdm_model = cmdstanpy.CmdStanModel(model_name="ANN-RDM_full_FC",
                                   stan_file=stan_file_path)

iteration_count = 0
params_range = pd.read_csv("Data/params_range.csv", index_col=0)
while iteration_count < number_of_participants:
    print(f"Iteration for participant {iteration_count+1} Started")
    parameters_set = params_range.copy()
    parameters_set["generated"] = np.random.normal(loc=parameters_set.iloc[:, 0],
                                                   scale=parameters_set.iloc[:, 1])

    behavioral_df = simulate_ANNRDM_individual(n_trials=n_trials, trials_info_df=word_nword_df,
                                         parameters_set=parameters_set)
    stan_parameters = get_stan_parameters(behavioral_df)
    fit = rdm_model.sample(data=stan_parameters,
                       iter_sampling=n_sample, 
                       iter_warmup=n_warmup,
                       chains=n_chains,
                       show_console=False)
    
    df = fit.summary()
    badRhat = False
    for f in df["R_hat"]:
        if f >= 1.01 or f <= 0.9:
            badRhat = True
    
    if badRhat:
        print("Split R-hat values are not satisfactory for all parameters. repeating iteration") 
    else:
        save_results_to_csv(fit, parameters_set)
        print(f"Iteration for participant {iteration_count+1} Finished") 
        iteration_count += 1


## Particpants parameter recovery
recovery_data = pd.read_csv("RecoveryResults/threshold_nonword_recovery_results.csv", header=0)

fig, ax = plt.subplots(1, 1, figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
posterior_mean = recovery_data["mean_threshold_nonword"]
posterior_median = recovery_data["median_threshold_nonword"]
true = recovery_data["real_threshold_nonword"]
ax.scatter(true, posterior_mean, color="tomato",
                                zorder=10)
ax.scatter(true, posterior_median, color="yellow",
                                zorder=9)
ax.vlines(x=true.to_numpy(), linewidth=2,
          ymin=recovery_data["HDI_threshold_nonword_bottom"].to_numpy(),
          ymax=recovery_data["HDI_threshold_nonword_top"].to_numpy())
ax.set_title("threshold_nonword")

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
plt.savefig("Plots/threshold_nonword_recovery.pdf")

# ndt
stan_file_path = root +  "models/stan/ANN-RDM/individual/ndt_recovery.stan" 

def save_results_to_csv(fit, parameters_set):
    columns = {}
    
    recoverd_df = pd.DataFrame([], columns=columns.keys())
    stan_variables = fit.stan_variables()

    columns["mean_ndt"] = stan_variables["ndt"].mean()
    columns["median_ndt"] = np.median(stan_variables["ndt"])
    columns["real_ndt"] = parameters_set.loc["ndt", "generated"]
    columns["HDI_ndt_bottom"], columns["HDI_ndt_top"] = hdi(stan_variables["ndt"])

    
    recoverd_df = pd.concat([recoverd_df, pd.DataFrame(columns, index=[0])])

    output_path="RecoveryResults/ndt_recovery_results.csv"
    recoverd_df.to_csv(output_path,
                       mode="a",
                       header=not os.path.exists(output_path),
                       index=False)


## Simulation and Estimation process
rdm_model = cmdstanpy.CmdStanModel(model_name="ANN-RDM_full_FC",
                                   stan_file=stan_file_path)

iteration_count = 0
params_range = pd.read_csv("Data/params_range.csv", index_col=0)
while iteration_count < number_of_participants:
    print(f"Iteration for participant {iteration_count+1} Started")
    parameters_set = params_range.copy()
    parameters_set["generated"] = np.random.normal(loc=parameters_set.iloc[:, 0],
                                                   scale=parameters_set.iloc[:, 1])

    behavioral_df = simulate_ANNRDM_individual(n_trials=n_trials, trials_info_df=word_nword_df,
                                         parameters_set=parameters_set)
    stan_parameters = get_stan_parameters(behavioral_df)
    fit = rdm_model.sample(data=stan_parameters,
                       iter_sampling=n_sample, 
                       iter_warmup=n_warmup,
                       chains=n_chains,
                       show_console=False)
    parameters_set.loc["ndt"] = behavioral_df.loc[0, "ndt"]
    
    df = fit.summary()
    badRhat = False
    for f in df["R_hat"]:
        if f >= 1.01 or f <= 0.9:
            badRhat = True
    
    if badRhat:
        print("Split R-hat values are not satisfactory for all parameters. repeating iteration") 
    else:
        save_results_to_csv(fit, parameters_set)
        print(f"Iteration for participant {iteration_count+1} Finished") 
        iteration_count += 1
        
## Particpants parameter recovery
recovery_data = pd.read_csv("RecoveryResults/ndt_recovery_results.csv", header=0)

fig, ax = plt.subplots(1, 1, figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
posterior_mean = recovery_data["mean_ndt"]
posterior_median = recovery_data["median_ndt"]
true = recovery_data["real_ndt"]
ax.scatter(true, posterior_mean, color="tomato",
                                zorder=10)
ax.scatter(true, posterior_median, color="yellow",
                                zorder=9)
ax.vlines(x=true.to_numpy(), linewidth=2,
          ymin=recovery_data["HDI_ndt_bottom"].to_numpy(),
          ymax=recovery_data["HDI_ndt_top"].to_numpy())
ax.set_title("ndt")

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
plt.savefig("Plots/ndt_recovery.pdf")