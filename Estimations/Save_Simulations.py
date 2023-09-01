# Importing Packages
import sys
import gc
import os
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

import numpy as np
import pandas as pd
import cmdstanpy
from utils.random import random_rdm_2A


def simulate_models(estimated_models_path):
    for path in estimated_models_path:
        model_name = path.split('/')[-1]
        print(f'Starting Simultion for {model_name}')
        
        # Loading Model
        fit = cmdstanpy.from_csv(path)

        drift_word_t = fit.stan_variables()["drift_word_t"]
        drift_nonword_t = fit.stan_variables()["drift_nonword_t"]
        threshold_t_word = fit.stan_variables()["threshold_t_word"]
        threshold_t_nonword = fit.stan_variables()["threshold_t_nonword"]
        ndt_t = fit.stan_variables()["ndt_t"]

        # Simulating model with estimated parameters
        pp_rt, pp_response = random_rdm_2A(drift_word_t, drift_nonword_t,
                                           threshold_t_word, threshold_t_nonword, ndt_t,
                                           noise_constant=1, dt=0.001, max_rt=5)
        
        # Save RT and Response arrays
        with open(f'Results/Simulations/{model_name}.npy', 'wb') as f:
            np.save(f, pp_rt)
            np.save(f, pp_response)
        
        print(f'Simultion of {model_name} compeleted')
        
        # freeing for next iteration RAM
        del fit, pp_rt, pp_response
        gc.collect()


models_path = ['Results/hierarchical/stan_results/ANN-RDM_s_am_FT', 'Results/hierarchical/stan_results/RDM_cd'] 
simulate_models(models_path)