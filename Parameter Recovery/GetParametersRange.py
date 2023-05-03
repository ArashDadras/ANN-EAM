import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

from utils.utils import get_parameters_range, hdi

stan_output_dir = "../Estimations/Results/hierarchical/stan_results/ANN-RDM_full_FT"
# DONT CHANGE THE ORDER
needed_parameters = ["alpha_sbj", "b_sbj", "k_1_sbj", "k_2_sbj",
                    "threshold_sbj_word", "threshold_sbj_nonword",
                    "g_sbj", "m_sbj"]
params_range = get_parameters_range(stan_output_dir, needed_parameters)
params_range.to_csv("params_range.csv")