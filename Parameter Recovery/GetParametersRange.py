import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

import numpy as np
import pandas as pd
import cmdstanpy
from utils.utils import get_parameters_range

# Loading words and non-words with zipf and predicted probabilities
root = "../"
plots_root = "Results/hierarchical/Plots/"
datasets_root = root + "Datasets/"
behavioural_data_root = datasets_root +  "behavioral_data/selected_data/"
dataset_path = datasets_root + "AI Models Results/fastText_FC.csv"
path_to_stan_output = root + "Estimations/Results/hierarchical/stan_results/ANN-RDM_full_FT"


word_nword_df = pd.read_csv(dataset_path, header=None,
                            names =["string", "freq",  "label", "zipf",
                                    "category", "word_prob", "non_word_prob"])
word_nword_df

# Reading LDT Data
behavioural_df = pd.read_csv(behavioural_data_root + "LDT_data.csv",
                             header=None,
                             names=["accuracy", "rt", "string", "response",
                                    "participant", "minRT", "participant_id"])
# Merging  behavioral dataframe with word_nonword_df to have words and non-words data with behavioral data
behavioural_df = pd.merge(behavioural_df, word_nword_df, on="string", how="left").dropna().reset_index(drop=True)
behavioural_df = behavioural_df.drop(["freq", "participant"], axis=1)

ranges = get_parameters_range(path_to_stan_output, behavioural_df)
ranges.index = ["alpha", "b", "k_1", "k_2", "threshold_word", "threshold_nonword", "g", "m"]
ranges.to_csv("Data/params_range.csv")