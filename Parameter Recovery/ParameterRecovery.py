import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
import numpy as np
import pandas as pd
import cmdstanpy
from utils.random import simulate_ANNRDM_individual
from utils.utils import get_parameters_range, get_stan_parameters, save_results_to_csv, plot_parameter_recovery_results

root = ".."
behavioural_data_root = f"{root}/Datasets/behavioral_data/selected_data/" 
dataset_path = f"{root}/Datasets/AI Models Results/fastText_FC.csv"

word_nword_df = pd.read_csv(dataset_path, header=None,
                            names =["string", "freq",  "label", "zipf",
                                    "category", "word_prob", "non_word_prob"])

number_of_participants = 100
n_trials = 400

# set sampling parameters
n_iter = 8000
n_warmup = int(n_iter/2)
n_sample = int(n_iter/2)
n_chains = 4

priors = {
    'threshold_priors': [2, 1],
    'ndt_priors': [0, 1],
    'alpha_priors': [0, 1],
    'b_priors': [0, 1],
    'k_priors': [2, 1] 
}

params_range = pd.read_csv("Data/params_range.csv", index_col=0)

def paremeter_recovery(params=['k_1', 'k_2', 'b', 'alpha', 'threshold_word', 'threshold_nonword', 'ndt']):
    for param in params:
        print(f"Starting recovery for {param} parameter")
        
        stan_file_path = f'{root}/models/stan/ANN-RDM/individual/{param}_recovery.stan'
        model_name = f'{param}_pr'
        
        rdm_model = cmdstanpy.CmdStanModel(model_name=model_name,
                                   stan_file=stan_file_path)
        
        iteration_count = 0
        while iteration_count < number_of_participants:
            print(f"Iteration for participant {iteration_count+1} Started")
            parameters_set = params_range.copy()
            parameters_set["generated"] = np.random.normal(loc=parameters_set.iloc[:, 0],
                                                           scale=parameters_set.iloc[:, 1])

            behavioral_df = simulate_ANNRDM_individual(n_trials=n_trials, trials_info_df=word_nword_df,
                                                 parameters_set=parameters_set)
            stan_parameters = get_stan_parameters(behavioral_df, priors)
            try:
                fit = rdm_model.sample(data=stan_parameters,
                                   iter_sampling=n_sample, 
                                   iter_warmup=n_warmup,
                                   chains=n_chains,
                                   show_console=False)
            except:
                print("Could not fit model becuase of generated parameters")
                continue

            print("dasdsadd")
            df = fit.summary()
            badRhat = False
            for f in df["R_hat"]:
                if f >= 1.01 or f <= 0.9:
                    badRhat = True

            if badRhat:
                print("Split R-hat values are not satisfactory for all parameters. repeating iteration") 
            else:
                save_results_to_csv(fit, parameters_set, param)
                print(f"Iteration for participant {iteration_count+1} Finished") 
                iteration_count += 1
                
        plot_parameter_recovery_results(param_name=param)
        
        print(f"Finished recovery for {param} parameter")
        
paremeter_recovery()