# Importing Packages
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
import json
from utils.random import random_rdm_2A
from utils.utils import get_dfs, calculate_waic, bci, plot_mean_posterior
import argparse

# Set plt configs
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['pdf.use14corefonts'] = True


# Choose Model
# Setting agrparse
parser = argparse.ArgumentParser(
    description='Estimates RDM (ANN-RDM) parameters')
parser.add_argument('model_name', metavar='N',
                    help='Model Name for estimation')

args = parser.parse_args()
print("Model name: " + args.model_name)

# roots
root = "../"
plots_root = "Results/hierarchical/Plots/"
datasets_root = root + "Datasets/"
behavioural_data_root = datasets_root +  "behavioral_data/selected_data/" 
stan_files_root = root +  "models/stan/" 
saved_models_root = "Results/hierarchical/stan_results/"

model_config = {}
plots_path = ""
dataset_path = ""
stan_file_path = ""
stan_output_dir = ""

# read models configuration json file
with open("../models/rdm_based_models.json") as f:
    models = json.load(f)
    models_name = list(models.keys())

if not args.model_name in models_name:
    sys.exit("Not a valid model")
    
# Choose and set model configuration
def SetModelAndPaths(model_name):
    global model_config
    global plots_path
    global dataset_path
    global stan_file_path
    global stan_output_dir
    model_config = models[model_name]
    plots_path = plots_root + model_config["plots_folder_name"] + "/"
    dataset_path = datasets_root + "AI Models Results/" + model_config["dataset_name"]
    stan_file_path = stan_files_root + model_config["stan_file"]
    stan_output_dir = saved_models_root + model_config["model_name"] + "/"
    
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
        print("Directory " , plots_path ,  " Created ")
    else:    
        print("Directory " , plots_path ,  " already exists")
        
    if not os.path.exists(stan_output_dir):
        os.makedirs(stan_output_dir)
        print("Directory " , stan_output_dir ,  " Created ")
    else:    
        print("Directory " , stan_output_dir ,  " already exists")


SetModelAndPaths(args.model_name)

print("Model's configs:")
print(model_config)

# Prepare data
# Loading words and non-words with zipf and predicted probabilities
word_nword_df = pd.read_csv(dataset_path, header=None,
                            names =["string", "freq",  "label", "zipf",
                                    "category", "word_prob", "non_word_prob"])

# Reading LDT Data
behavioural_df = pd.read_csv(behavioural_data_root + "LDT_data.csv",
                             header=None,
                             names=["accuracy", "rt", "string", "response",
                                    "participant", "minRT", "participant_id"])
# Merging  behavioral dataframe with word_nonword_df to have words and non-words data with behavioral data
behavioural_df = pd.merge(behavioural_df, word_nword_df, on="string", how="left").dropna().reset_index(drop=True)
behavioural_df = behavioural_df.drop(["freq", "participant"], axis=1)

# Stan Model and Estimation
# Compiling stan model
rdm_model = cmdstanpy.CmdStanModel(model_name=model_config['model_name'],
                                   stan_file=stan_file_path)
# Preparing model's inputs
# note that some inputs of data_dict might not be used depending on which model is used
# For all models
N = len(behavioural_df)                                                    # For all models
participant = behavioural_df["participant_id"].to_numpy()                     # For all models
p = behavioural_df.loc[:, ["word_prob", "non_word_prob"]].to_numpy()       # predicted probabilites of words and non-words, for ANN-EAM models
frequency = behavioural_df["zipf"].to_numpy().astype(int)                  # zipf values, for models with non-decision time or drift modulation
frequencyCondition = behavioural_df["category"].replace(["HF", "LF", "NW"], [1, 2, 3]).to_numpy() # For models with conditional drift
response = behavioural_df["response"].to_numpy().astype(int)               # for all models
rt = behavioural_df["rt"].to_numpy()                                       # for all models
minRT = behavioural_df["minRT"].to_numpy()                                 # for all models
RTbound = 0.1                                                              # for all models
Number_Of_Participants = len(set(behavioural_df["participant_id"]))

threshold_priors = [2, 1, 1, 1]          # For all models with RDM
ndt_priors = [0, 0.5, 1, 1];             # For models wtihout non-decision time modulation
g_priors = [-2, 1, 0, 1]                 # For models wtih non-decision time modulation
m_priors = [0, 0.5, 0, 1]                # For models wtih non-decision time modulation
drift_priors = [1, 2, 1, 1]              # For models without drift mapping functions (non ANN-EAM models)
alpha_priors = [0, 1, 1, 1]              # For models with drift mapping functions
b_priors = [0, 1, 1, 1]                  # For models with drift mapping functions with asymptote modulation and linear models
k_priors = [2, 1, 1, 1]                  # For models with sigmoid drift mapping functions (ANN-EAM models)

# define input for the model
data_dict = {"N": N,
             "L": Number_Of_Participants,
             "participant": participant,
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
             }

# set sampling parameters
n_iter = 9000
n_warmup = int(n_iter/2)
n_sample = int(n_iter/2)
n_chains = 4

# Fitting the model
fit = rdm_model.sample(data=data_dict,
                       iter_sampling=n_sample, 
                       iter_warmup=n_warmup,
                       chains=n_chains,
                       output_dir=stan_output_dir,
                       show_console=True)

# Model diagnostics
print("***hmc diagnostics:")
print(fit.diagnose(), flush=True)

df = fit.summary()

print("***DF: ")
print(df)

counter = 0
print("***Rhat > 1.01: ")
for f in df["R_hat"]:
    if f >= 1.01 or f <= 0.9:
        counter += 1
print(counter)

df.loc[df["R_hat"]>1.01].to_csv("Results/hierarchical/logs/" + model_config["model_name"] + "_rhat_log.csv")

print(df.loc[df['R_hat'] > 1.01])

print(df.loc[df['R_hat'] > 1.01].describe())

# Check parameters
# Parameters posterior plots
az.plot_posterior(fit, var_names=model_config["transf_params"],
                  hdi_prob=.95);
plt.savefig(plots_path + "Parameters.pdf")

# Loading model parameters for each trial
drift_word_t = fit.stan_variables()['drift_word_t']
drift_nonword_t = fit.stan_variables()['drift_nonword_t']
if model_config['model_name'] != "RDM":
    threshold_t_word = fit.stan_variables()['threshold_t_word']
    threshold_t_nonword = fit.stan_variables()['threshold_t_nonword']
else:
    threshold_t = fit.stan_variables()['threshold_t']
ndt_t = fit.stan_variables()['ndt_t']

# Models mean parameters in different conditions
HF_condition_w = drift_word_t[:, behavioural_df['category'] == "HF"]
HF_condition_nw = drift_nonword_t[:, behavioural_df['category'] == "HF"]
LF_condition_w = drift_word_t[:, behavioural_df['category'] == "LF"]
LF_condition_nw = drift_nonword_t[:, behavioural_df['category'] == "LF"]
NW_condition_w = drift_word_t[:, behavioural_df['category'] == "NW"]
NW_condition_nw = drift_nonword_t[:, behavioural_df['category'] == "NW"]

print('HF words, word drift mean and std:')
print(np.mean(np.mean(HF_condition_w, axis=1)),
      np.std(np.mean(HF_condition_w, axis=1)))
print('HF words, nonword drift mean and std:')
print(np.mean(np.mean(HF_condition_nw, axis=1)),
      np.std(np.mean(HF_condition_nw, axis=1)))
print('LF words word drift mean and std:')
print(np.mean(np.mean(LF_condition_w, axis=1)),
      np.std(np.mean(LF_condition_w, axis=1)))
print('LF words nonword drift mean and std:')
print(np.mean(np.mean(LF_condition_nw, axis=1)),
      np.std(np.mean(LF_condition_nw, axis=1)))
print('NW words word drift mean and std:')
print(np.mean(np.mean(NW_condition_w, axis=1)),
      np.std(np.mean(NW_condition_w, axis=1)))
print('NW words nonword drift mean and std:')
print(np.mean(np.mean(NW_condition_nw, axis=1)),
      np.std(np.mean(NW_condition_nw, axis=1)))

if model_config['model_name'] != "RDM":
    HF_condition_w = threshold_t_word[:, behavioural_df['category'] == "HF"]
    HF_condition_nw = threshold_t_nonword[:,
                                          behavioural_df['category'] == "HF"]
    LF_condition_w = threshold_t_word[:, behavioural_df['category'] == "LF"]
    LF_condition_nw = threshold_t_nonword[:,
                                          behavioural_df['category'] == "LF"]
    NW_condition_w = threshold_t_word[:, behavioural_df['category'] == "NW"]
    NW_condition_nw = threshold_t_nonword[:,
                                          behavioural_df['category'] == "NW"]
else:
    HF_condition = threshold_t[:, behavioural_df['category'] == "HF"]
    LF_condition = threshold_t[:, behavioural_df['category'] == "LF"]
    NW_condition = threshold_t[:, behavioural_df['category'] == "NW"]

if model_config['model_name'] != "RDM":
    print('HF words, word threshold mean and std:')
    print(np.mean(np.mean(HF_condition_w, axis=1)),
          np.std(np.mean(HF_condition_w, axis=1)))
    print('HF words, nonword threshold mean and std:')
    print(np.mean(np.mean(HF_condition_nw, axis=1)),
          np.std(np.mean(HF_condition_nw, axis=1)))
    print('LF words word threshold mean and std:')
    print(np.mean(np.mean(LF_condition_w, axis=1)),
          np.std(np.mean(LF_condition_w, axis=1)))
    print('LF words nonword threshold mean and std:')
    print(np.mean(np.mean(LF_condition_nw, axis=1)),
          np.std(np.mean(LF_condition_nw, axis=1)))
    print('NW words word threshold mean and std:')
    print(np.mean(np.mean(NW_condition_w, axis=1)),
          np.std(np.mean(NW_condition_w, axis=1)))
    print('NW words nonword threshold mean and std:')
    print(np.mean(np.mean(NW_condition_nw, axis=1)),
          np.std(np.mean(NW_condition_nw, axis=1)))
else:
    print('HF words, threshold mean and std:')
    print(np.mean(np.mean(HF_condition, axis=1)),
          np.std(np.mean(HF_condition, axis=1)))
    print('LF words, threshold mean and std:')
    print(np.mean(np.mean(LF_condition, axis=1)),
          np.std(np.mean(LF_condition, axis=1)))
    print('NW words, word threshold mean and std:')
    print(np.mean(np.mean(NW_condition, axis=1)),
          np.std(np.mean(NW_condition, axis=1)))

HF_condition = ndt_t[:, behavioural_df['category'] == "HF"]
LF_condition = ndt_t[:, behavioural_df['category'] == "LF"]
NW_condition = ndt_t[:, behavioural_df['category'] == "NW"]

print('HF words ndt_t mean and std:')
print(np.mean(np.mean(HF_condition, axis=1)),
      np.std(np.mean(HF_condition, axis=1)))
print('LF words ndt_t mean and std:')
print(np.mean(np.mean(LF_condition, axis=1)),
      np.std(np.mean(LF_condition, axis=1)))
print('Non Words ndt_t mean and std:')
print(np.mean(np.mean(NW_condition, axis=1)),
      np.std(np.mean(NW_condition, axis=1)))

# Calculating metrics
log_likelihood = fit.stan_variables()["log_lik"]
print("Estimation Fit metrics:")
print(calculate_waic(log_likelihood))

# Simulating RDM with estimated parameters
if model_config['model_name'] != "RDM":
    pp_rt, pp_response = random_rdm_2A(drift_word_t, drift_nonword_t,
                                       threshold_t_word, threshold_t_nonword, ndt_t,
                                       noise_constant=1, dt=0.001, max_rt=5)
else:
    pp_rt, pp_response = random_rdm_2A(drift_word_t, drift_nonword_t, threshold_t,
                                       threshold_t, ndt_t, noise_constant=1, dt=0.001,
                                       max_rt=5)

with open(f'Results/Simulations/{model_config["model_name"]}.npy', 'wb') as f:
    np.save(f, pp_rt)
    np.save(f, pp_response)

# Predicted Data
tmp1 = pd.DataFrame(pp_rt,
                    index=pd.Index(np.arange(1, len(pp_rt)+1), name="sample"),
                    columns=pd.MultiIndex.from_product((["rt"],
                                                        np.arange(pp_rt.shape[1])),
                                                        names=["variable", "trial"]))
tmp2 = pd.DataFrame(pp_response,
                    index=pd.Index(np.arange(1, len(pp_response)+1), name="sample"),
                    columns=pd.MultiIndex.from_product((["response"],
                                                        np.arange(pp_response.shape[1])),
                                                               names=["variable", "trial"]))
predictedData = pd.concat((tmp1, tmp2), axis=1)

# RT Quantiles Posterior Predictions Checks
quantiles = [.1, .3, .5, .7, .9]

# All Trials
exp_all_trials, pred_all_trials = get_dfs(behavioural_df, predictedData)

all_quantiles_ex = exp_all_trials["rt"].quantile(quantiles)
all_quantiles_pred = pred_all_trials.quantile(quantiles, axis=1).T
all_predicted_bci = np.array([bci(all_quantiles_pred[x]) for x in quantiles])

fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(wspace=0.2, hspace=0.5)

ax.set_title("All Trials quantiles", fontweight="bold", size=14)
ax.scatter(quantiles, all_quantiles_ex, color="black", s=100)

ax.fill_between(quantiles,
                all_predicted_bci[:, 0],
                all_predicted_bci[:, 1],
                all_predicted_bci[:, 0] < all_predicted_bci[:, 1],  color = "orange", alpha=0.5)

ax.set_xlabel("Quantiles", fontsize=14)
ax.set_xticks(quantiles)
ax.set_xticklabels(quantiles)
ax.set_ylabel("RT", fontsize=14)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(12) 

sns.despine()
plt.savefig(plots_path + "PPC-Quantiles-All Trials.pdf")

# All Trials (word response vs non-word response)
exp_word_resp_all, pred_word_resp_all = get_dfs(behavioural_df, predictedData,
                                                response=1)
exp_nonword_resp_all, pred_nonword_resp_all = get_dfs(behavioural_df, predictedData,
                                                      response=0)

word_quantiles_ex = exp_word_resp_all["rt"].quantile(quantiles)
nonword_quantiles_ex = exp_nonword_resp_all["rt"].quantile(quantiles)

word_quantiles_pred = pred_word_resp_all.quantile(quantiles, axis=1).T
nonword_quantiles_pred = pred_nonword_resp_all.quantile(quantiles, axis=1).T

word_predicted_bci = np.array([bci(word_quantiles_pred[x]) for x in quantiles])
nonword_predicted_bci = np.array([bci(nonword_quantiles_pred[x]) for x in quantiles])

fig, axes = plt.subplots(1, 2, figsize=(25,6))
plt.subplots_adjust(wspace=0.2, hspace=0.5)

axes[0].set_title("Word trials quantiles", fontweight="bold", size=20)
axes[1].set_title("Non Word trials quantiles", fontweight="bold", size=20)

axes[0].scatter(quantiles, word_quantiles_ex, color="black", s=100)
axes[1].scatter(quantiles, nonword_quantiles_ex, color="black", s=100)

axes[0].fill_between(quantiles,
                word_predicted_bci[:, 0],
                word_predicted_bci[:, 1],
                word_predicted_bci[:, 0] < word_predicted_bci[:, 1],  color = "tomato", alpha=0.5)

axes[1].fill_between(quantiles,
                nonword_predicted_bci[:, 0],
                nonword_predicted_bci[:, 1],
                nonword_predicted_bci[:, 0] < nonword_predicted_bci[:, 1],  color = "powderblue", alpha=0.5)

for ax in axes:
        ax.set_xlabel("Quantiles", fontsize=16)
        ax.set_xticks(quantiles)
        ax.set_xticklabels(quantiles)
        ax.set_ylabel("RT", fontsize=16)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(12) 

sns.despine()
plt.savefig(plots_path + "PPC-Quantiles-All Trials-Word vs Nonword.pdf")

# All trials (Correct Choice vs Incorrect Choice)
exp_cor_choice_all, _ = get_dfs(behavioural_df, predictedData,
                                accuracy=1)
exp_incor_resp_all, _ = get_dfs(behavioural_df, predictedData,
                                accuracy=0)
pred_cor_choice_all = predictedData["rt"][predictedData["response"]==behavioural_df["label"]]
pred_incor_choice_all = predictedData["rt"][predictedData["response"]!=behavioural_df["label"]]

cor_quantiles_ex = exp_cor_choice_all["rt"].quantile(quantiles)
incor_quantiles_ex = exp_incor_resp_all["rt"].quantile(quantiles)

cor_quantiles_pred = pred_cor_choice_all.quantile(quantiles, axis=1).T
incor_quantiles_pred = pred_incor_choice_all.quantile(quantiles, axis=1).T

cor_predicted_bci = np.array([bci(cor_quantiles_pred[x]) for x in quantiles])
incor_predicted_bci = np.array([bci(incor_quantiles_pred[x]) for x in quantiles])

fig, axes = plt.subplots(1, 2, figsize=(25,6))
plt.subplots_adjust(wspace=0.2, hspace=0.5)

axes[0].set_title("Correct choice quantiles", fontweight="bold", size=20)
axes[1].set_title("Incorrect choice quantiles", fontweight="bold", size=20)

axes[0].scatter(quantiles, cor_quantiles_ex, color="black", s=100)
axes[1].scatter(quantiles, incor_quantiles_ex, color="black", s=100)

axes[0].fill_between(quantiles,
                cor_predicted_bci[:, 0],
                cor_predicted_bci[:, 1],
                cor_predicted_bci[:, 0] < cor_predicted_bci[:, 1],  color = "coral", alpha=0.5)

axes[1].fill_between(quantiles,
                incor_predicted_bci[:, 0],
                incor_predicted_bci[:, 1],
                incor_predicted_bci[:, 0] < incor_predicted_bci[:, 1],  color = "palegreen", alpha=0.5)

for ax in axes:
        ax.set_xlabel("Quantiles", fontsize=14)
        ax.set_xticks(quantiles)
        ax.set_xticklabels(quantiles)
        ax.set_ylabel("RTs upper boundary", fontsize=14)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(12) 

sns.despine()
plt.savefig(plots_path + "PPC-Quantiles-All Trials-Correct vs Incorrect.pdf")

# Conditional (HF, LF, NW trials)
exp_HF_trials, pred_HF_trials = get_dfs(behavioural_df, predictedData,
                                        category="HF")
exp_LF_trials, pred_LF_trials = get_dfs(behavioural_df, predictedData,
                                        category="LF")
exp_NW_trials, pred_NW_trials = get_dfs(behavioural_df, predictedData,
                                        category="NW")

# experiment Data quantile
HF_quantile_ex = exp_HF_trials["rt"].quantile(quantiles)
LF_quantile_ex = exp_LF_trials["rt"].quantile(quantiles)
NW_quantile_ex = exp_NW_trials["rt"].quantile(quantiles)

# predicted data quantiles (for each sample)
HF_quantile_pred = pred_HF_trials.quantile(quantiles, axis=1).T
LF_quantile_pred = pred_LF_trials.quantile(quantiles, axis=1).T
NW_quantile_pred = pred_NW_trials.quantile(quantiles, axis=1).T

# predicted data quantiles bci
HF_predicted_bci = np.array([bci(HF_quantile_pred[x]) for x in quantiles])
LF_predicted_bci = np.array([bci(LF_quantile_pred[x]) for x in quantiles])
NW_predicted_bci = np.array([bci(NW_quantile_pred[x]) for x in quantiles])

fig, axes = plt.subplots(1,3 , figsize=(25,5))
plt.subplots_adjust(wspace=0.1, hspace=0.5)

axes[0].set_title("HF quantiles", fontweight="bold", size=20)
axes[1].set_title("LF quantiles", fontweight="bold", size=20)
axes[2].set_title("NW quantiles", fontweight="bold", size=20)

axes[0].scatter(quantiles, HF_quantile_ex, color="black", s=100)
axes[1].scatter(quantiles, LF_quantile_ex, color="black", s=100)
axes[2].scatter(quantiles, NW_quantile_ex, color="black", s=100)

axes[0].fill_between(quantiles,
                HF_predicted_bci[:, 0],
                HF_predicted_bci[:, 1],
                HF_predicted_bci[:, 0] < HF_predicted_bci[:, 1],  color = "gold", alpha=0.3)

axes[1].fill_between(quantiles,
                LF_predicted_bci[:, 0],
                LF_predicted_bci[:, 1],
                LF_predicted_bci[:, 0] < LF_predicted_bci[:, 1],  color = "lightskyblue", alpha=0.3)

axes[2].fill_between(quantiles,
                NW_predicted_bci[:, 0],
                NW_predicted_bci[:, 1],
                NW_predicted_bci[:, 0] < NW_predicted_bci[:, 1],  color = "limegreen", alpha=0.3)


for ax in axes:
        ax.set_xlabel("Quantiles", fontsize=18)
        ax.set_xticks(quantiles)
        ax.set_xticklabels(quantiles)
        ax.set_ylabel("RTs upper boundary", fontsize=18)
        for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(13)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(13) 

sns.despine()
plt.savefig(plots_path + "PPC-Quantiles-Conditional.pdf")

# Conditional (HF, LF, NW trials) for word response and nonword response
exp_word_resp_HF, pred_word_resp_HF = get_dfs(behavioural_df, predictedData,
                                              category="HF", response=1)
exp_word_resp_LF, pred_word_resp_LF = get_dfs(behavioural_df, predictedData,
                                              category="LF", response=1)
exp_word_resp_NW, pred_word_resp_NW = get_dfs(behavioural_df, predictedData,
                                              category="NW", response=1)

exp_nonword_resp_HF, pred_nonword_resp_HF = get_dfs(behavioural_df, predictedData,
                                                    category="HF", response=0)
exp_nonword_resp_LF, pred_nonword_resp_LF = get_dfs(behavioural_df, predictedData,
                                                    category="LF", response=0)
exp_nonword_resp_NW, pred_nonword_resp_NW = get_dfs(behavioural_df, predictedData,
                                                    category="NW", response=0)

# experiment Data quantile
HF_word_quantile_ex = exp_word_resp_HF["rt"].quantile(quantiles)
LF_word_quantile_ex = exp_word_resp_LF["rt"].quantile(quantiles)
NW_word_quantile_ex = exp_word_resp_NW["rt"].quantile(quantiles)

HF_nonword_quantile_ex = exp_nonword_resp_HF["rt"].quantile(quantiles)
LF_nonword_quantile_ex = exp_nonword_resp_LF["rt"].quantile(quantiles)
NW_nonword_quantile_ex = exp_nonword_resp_NW["rt"].quantile(quantiles)

# predicted data quantiles (for each sample)
HF_word_quantile_pred = pred_word_resp_HF.quantile(quantiles, axis=1).T
LF_word_quantile_pred = pred_word_resp_LF.quantile(quantiles, axis=1).T
NW_word_quantile_pred = pred_word_resp_NW.quantile(quantiles, axis=1).T

HF_nonword_quantile_pred = pred_nonword_resp_HF.quantile(quantiles, axis=1).T
LF_nonword_quantile_pred = pred_nonword_resp_LF.quantile(quantiles, axis=1).T
NW_nonword_quantile_pred = pred_nonword_resp_NW.quantile(quantiles, axis=1).T


# predicted data quantiles bci
HF_word_predicted_bci = np.array([bci(HF_word_quantile_pred[x]) for x in quantiles])
LF_word_predicted_bci = np.array([bci(LF_word_quantile_pred[x]) for x in quantiles])
NW_word_predicted_bci = np.array([bci(NW_word_quantile_pred[x]) for x in quantiles])

HF_nonword_predicted_bci = np.array([bci(HF_nonword_quantile_pred[x]) for x in quantiles])
LF_nonword_predicted_bci = np.array([bci(LF_nonword_quantile_pred[x]) for x in quantiles])
NW_nonword_predicted_bci = np.array([bci(NW_nonword_quantile_pred[x]) for x in quantiles])

fig, axes = plt.subplots(2,3 , figsize=(30,10))
plt.subplots_adjust(wspace=0.2, hspace=0.5)

axes[0][0].set_title("HF quantiles word choice", fontweight="bold", size=20)
axes[0][1].set_title("LF quantiles word choice", fontweight="bold", size=20)
axes[0][2].set_title("NW quantiles word choice", fontweight="bold", size=20)

axes[1][0].set_title("HF quantiles non-word choice", fontweight="bold", size=20)
axes[1][1].set_title("LF quantiles non-word choice", fontweight="bold", size=20)
axes[1][2].set_title("NW quantiles non-word choice", fontweight="bold", size=20)

axes[0][0].scatter(quantiles, HF_word_quantile_ex, color="black", s=90)
axes[0][1].scatter(quantiles, LF_word_quantile_ex, color="black", s=90)
axes[0][2].scatter(quantiles, NW_word_quantile_ex, color="black", s=90)

axes[1][0].scatter(quantiles, HF_nonword_quantile_ex, color="black", s=90)
axes[1][1].scatter(quantiles, LF_nonword_quantile_ex, color="black", s=90)
axes[1][2].scatter(quantiles, NW_nonword_quantile_ex, color="black", s=90)


axes[0][0].fill_between(quantiles,
                HF_word_predicted_bci[:, 0],
                HF_word_predicted_bci[:, 1],
                HF_word_predicted_bci[:, 0] < HF_word_predicted_bci[:, 1],  color = "gold", alpha=0.3)

axes[0][1].fill_between(quantiles,
                LF_word_predicted_bci[:, 0],
                LF_word_predicted_bci[:, 1],
                LF_word_predicted_bci[:, 0] < LF_word_predicted_bci[:, 1],  color = "lightskyblue", alpha=0.3)

axes[0][2].fill_between(quantiles,
                NW_word_predicted_bci[:, 0],
                NW_word_predicted_bci[:, 1],
                NW_word_predicted_bci[:, 0] < NW_word_predicted_bci[:, 1],  color = "limegreen", alpha=0.3)


axes[1][0].fill_between(quantiles,
                HF_nonword_predicted_bci[:, 0],
                HF_nonword_predicted_bci[:, 1],
                HF_nonword_predicted_bci[:, 0] < HF_nonword_predicted_bci[:, 1],  color = "gold", alpha=0.3)

axes[1][1].fill_between(quantiles,
                LF_nonword_predicted_bci[:, 0],
                LF_nonword_predicted_bci[:, 1],
                LF_nonword_predicted_bci[:, 0] < LF_nonword_predicted_bci[:, 1],  color = "lightskyblue", alpha=0.3)

axes[1][2].fill_between(quantiles,
                NW_nonword_predicted_bci[:, 0],
                NW_nonword_predicted_bci[:, 1],
                NW_nonword_predicted_bci[:, 0] < NW_nonword_predicted_bci[:, 1],  color = "limegreen", alpha=0.3)


for ax_d1 in axes:
    for ax in ax_d1:
        ax.set_xlabel("Quantiles", fontsize=18)
        ax.set_xticks(quantiles)
        ax.set_xticklabels(quantiles)
        ax.set_ylabel("RTs upper boundary", fontsize=18)
        for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(14)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14) 

sns.despine()
plt.savefig(plots_path + "PPC-Quantiles-Conditional-Word vs Nonword.pdf")

# Mean Accuracy and RT Posterior Prediction Checks
# All trials
exp_all_trials_rt, pred_all_trials_rt = get_dfs(behavioural_df, predictedData)
exp_all_trials_resp, pred_all_trials_resp = get_dfs(behavioural_df, predictedData,
                                                    pred_df_type="response")

all_data_rt_mean = exp_all_trials_rt["rt"].mean()
all_pred_rt_mean = pred_all_trials_rt.mean(axis=1)

all_data_resp_mean = exp_all_trials_resp["response"].mean()
all_pred_resp_mean = pred_all_trials_resp.mean(axis=1)

fig, axes = plt.subplots(1,2 , figsize=(12, 4))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

axes[0].set_title("All trials mean RT", fontweight="bold", size=16)
axes[1].set_title("All trials mean Response", fontweight="bold", size=16)

plot_mean_posterior(all_pred_rt_mean, all_data_rt_mean, axes[0])
plot_mean_posterior(all_pred_resp_mean, all_data_resp_mean, axes[1])

axes[0].set_xlabel("RT", fontsize=16)
axes[1].set_xlabel("Response", fontsize=16)

for ax in axes:
        ax.set_ylabel("Density", fontsize=16)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
            
plt.savefig(plots_path + "PPC-Mean Accuracy and RT-All trials.pdf")

# All Trials (correct choice vs incorrect choice)
exp_cor_all_trials_rt, pred_cor_all_trials_rt = get_dfs(behavioural_df, predictedData,
                                                        accuracy=1)
exp_incor_all_trials_rt, pred_incor_all_trials_rt = get_dfs(behavioural_df, predictedData,
                                                            accuracy=0)

exp_cor_all_trials_resp, pred_cor_all_trials_resp = get_dfs(behavioural_df, predictedData,
                                                            accuracy=1, pred_df_type="response")
exp_incor_all_trials_resp, pred_incor_all_trials_resp = get_dfs(behavioural_df, predictedData,
                                                                accuracy=0, pred_df_type="response")

all_trials_cor_rt_mean = exp_cor_all_trials_rt["rt"].mean()
all_pred_cor_rt_mean = pred_cor_all_trials_rt.mean(axis=1)

all_trials_incor_rt_mean = exp_incor_all_trials_rt["rt"].mean()
all_pred_incor_rt_mean = pred_incor_all_trials_rt.mean(axis=1)


all_data_cor_resp_mean = exp_cor_all_trials_resp["response"].mean()
all_pred_cor_resp_mean = pred_cor_all_trials_resp.mean(axis=1)

all_data_incor_resp_mean = exp_incor_all_trials_resp["response"].mean()
all_pred_incor_resp_mean = pred_incor_all_trials_resp.mean(axis=1)

fig, axes = plt.subplots(2,2 , figsize=(12,10))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

axes[0][0].set_title("Correct trials mean RT", fontweight="bold", size=16)
axes[0][1].set_title("Correct trials mean Response", fontweight="bold", size=16)
axes[1][0].set_title("Incorrect trials mean RT", fontweight="bold", size=16)
axes[1][1].set_title("Incorrect trials mean Response", fontweight="bold", size=16)

plot_mean_posterior(all_pred_cor_rt_mean, all_trials_cor_rt_mean, axes[0][0])
plot_mean_posterior(all_pred_cor_resp_mean, all_data_cor_resp_mean, axes[0][1])

plot_mean_posterior(all_pred_incor_rt_mean, all_trials_incor_rt_mean, axes[1][0])
plot_mean_posterior(all_pred_incor_resp_mean, all_data_incor_resp_mean, axes[1][1])

for ax in axes:
        ax[0].set_xlabel("RT", fontsize=15)
        ax[1].set_xlabel("Accuracy", fontsize=15)
        ax[0].set_ylabel("Density", fontsize=15)
        ax[1].set_ylabel("Density", fontsize=15)
        for tick in ax[0].xaxis.get_major_ticks():
            tick.label1.set_fontsize(13)
        for tick in ax[0].yaxis.get_major_ticks():
            tick.label1.set_fontsize(13)
        for tick in ax[1].xaxis.get_major_ticks():
            tick.label1.set_fontsize(13)
        for tick in ax[1].yaxis.get_major_ticks():
            tick.label1.set_fontsize(13) 

plt.savefig(plots_path + "PPC-Mean Accuracy and RT-All trials-Correct vs Incorrect.pdf")

# Conditional (HF, LF, NW trials)
exp_HF_trials_rt, pred_HF_trials_rt = get_dfs(behavioural_df, predictedData,
                                              category="HF")
exp_LF_trials_rt, pred_LF_trials_rt = get_dfs(behavioural_df, predictedData,
                                              category="LF")
exp_NW_trials_rt, pred_NW_trials_rt = get_dfs(behavioural_df, predictedData,
                                              category="NW")

exp_HF_trials_resp, pred_HF_trials_resp = get_dfs(behavioural_df, predictedData,
                                                  category="HF", pred_df_type="response")
exp_LF_trials_resp, pred_LF_trials_resp = get_dfs(behavioural_df, predictedData,
                                                  category="LF", pred_df_type="response")
exp_NW_trials_resp, pred_NW_trials_resp = get_dfs(behavioural_df, predictedData,
                                                  category="NW", pred_df_type="response")

HF_data_rt_mean = exp_HF_trials_rt["rt"].mean()
LF_data_rt_mean = exp_LF_trials_rt["rt"].mean()
NW_data_rt_mean = exp_NW_trials_rt["rt"].mean()

HF_pred_rt_mean = pred_HF_trials_rt.mean(axis=1)
LF_pred_rt_mean = pred_LF_trials_rt.mean(axis=1)
NW_pred_rt_mean = pred_NW_trials_rt.mean(axis=1)


HF_data_resp_mean = exp_HF_trials_resp["response"].mean()
LF_data_resp_mean = exp_LF_trials_resp["response"].mean()
NW_data_resp_mean = exp_NW_trials_resp["response"].mean()

HF_pred_resp_mean = pred_HF_trials_resp.mean(axis=1)
LF_pred_resp_mean = pred_LF_trials_resp.mean(axis=1)
NW_pred_resp_mean = pred_NW_trials_resp.mean(axis=1)

fig, axes = plt.subplots(3,2 , figsize=(12,15))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

axes[0][0].set_title("HF mean RT", fontweight="bold", size=16)
axes[0][1].set_title("HF mean Response", fontweight="bold", size=16)
axes[1][0].set_title("LF mean RT", fontweight="bold", size=16)
axes[1][1].set_title("LF mean Response", fontweight="bold", size=16)
axes[2][0].set_title("NW mean RT", fontweight="bold", size=16)
axes[2][1].set_title("NW mean Response", fontweight="bold", size=16)

plot_mean_posterior(HF_pred_rt_mean, HF_data_rt_mean, axes[0][0])
plot_mean_posterior(HF_pred_resp_mean, HF_data_resp_mean, axes[0][1])

plot_mean_posterior(LF_pred_rt_mean, LF_data_rt_mean, axes[1][0])
plot_mean_posterior(LF_pred_resp_mean, LF_data_resp_mean, axes[1][1])

plot_mean_posterior(NW_pred_rt_mean, NW_data_rt_mean, axes[2][0])
plot_mean_posterior(NW_pred_resp_mean, NW_data_resp_mean, axes[2][1])

for ax in axes:
        ax[0].set_xlabel("RT", fontsize=15)
        ax[1].set_xlabel("Accuracy", fontsize=15)
        ax[0].set_ylabel("Density", fontsize=15)
        ax[1].set_ylabel("Density", fontsize=15)
        for tick in ax[0].xaxis.get_major_ticks():
            tick.label1.set_fontsize(13)
        for tick in ax[0].yaxis.get_major_ticks():
            tick.label1.set_fontsize(13)
        for tick in ax[1].xaxis.get_major_ticks():
            tick.label1.set_fontsize(13)
        for tick in ax[1].yaxis.get_major_ticks():
            tick.label1.set_fontsize(13) 

plt.savefig(plots_path + "PPC-Mean Accuracy and RT-Conditional.pdf")
