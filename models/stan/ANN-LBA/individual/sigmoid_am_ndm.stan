// ANN-LBA with sigmoid mapping function for drifts with asymptote modulation,
// different threshold for words and nonwords,
// with non-decision time modulation

functions{

    // t, The decision time
    // b, the decision threshold
    // A, the maximum starting evidence,
    // v the drift rate
    // s, the standard deviation
    
    real lba_pdf(real t, real b, real A, real v, real s){
        //PDF of the LBA model
        real b_A_tv_ts;
        real b_tv_ts;
        real term_1;
        real term_2;
        real term_3;
        real term_4;
        real pdf;

        b_A_tv_ts = (b - A - t*v)/(t*s);
        b_tv_ts = (b - t*v)/(t*s);
        term_1 = v*Phi(b_A_tv_ts);
        term_2 = s*exp(normal_log(b_A_tv_ts,0,1));
        term_3 = v*Phi(b_tv_ts);
        term_4 = s*exp(normal_log(b_tv_ts,0,1));
        pdf = (1/A)*(-term_1 + term_2 + term_3 - term_4);

        return pdf;
     }

    real lba_cdf(real t, real b, real A, real v, real s){
        //CDF of the LBA model

        real b_A_tv;
        real b_tv;
        real ts;
        real term_1;
        real term_2;
        real term_3;
        real term_4;
        real cdf;

        b_A_tv = b - A - t*v;
        b_tv = b - t*v;
        ts = t*s;
        term_1 = b_A_tv/A * Phi(b_A_tv/ts);
        term_2 = b_tv/A   * Phi(b_tv/ts);
        term_3 = ts/A * exp(normal_log(b_A_tv/ts,0,1));
        term_4 = ts/A * exp(normal_log(b_tv/ts,0,1));
        cdf = 1 + term_1 - term_2 + term_3 - term_4;

        return cdf;

     }

    // word_threshold (b_word) = sp_trial_var_word + k_word 
    // nonword_threshold (b_nonword) = sp_trial_var_nonword + k_nonword
    real lba_lpdf(matrix RT, vector k_word, vector sp_trial_var_word,
                  vector k_nonword, vector sp_trial_var_nonword,
                  vector drift_word, vector drift_nonword,
                  vector ndt, vector s){

        real t;
        real b_word;
        real b_nonword;
        real cdf;
        real pdf;
        vector[rows(RT)] prob;
        real out;
        real prob_neg;

        for (i in 1:rows(RT)){
            b_word = sp_trial_var_word[i] + k_word[i];
            b_nonword = sp_trial_var_nonword[i] + k_nonword[i];            
            t = RT[i,1] - ndt[i];
            if(t > 0){
                cdf = 1;

                if(RT[i,2] == 1){
                    pdf = lba_pdf(t, b_word, sp_trial_var_word[i], drift_word[i], s[i]);
                    cdf = 1-lba_cdf(t, b_nonword, sp_trial_var_nonword[i], drift_nonword[i], s[i]);
                }
                else{
                    pdf = lba_pdf(t, b_nonword, sp_trial_var_nonword[i], drift_nonword[i], s[i]);
                    cdf = 1-lba_cdf(t, b_word, sp_trial_var_word[i], drift_word[i], s[i]);
                }
                prob_neg = Phi(-drift_word[i]/s[i]) * Phi(-drift_nonword[i]/s[i]);
                prob[i] = pdf*cdf;
                prob[i] = prob[i]/(1-prob_neg);
                if(prob[i] < 1e-10){
                    prob[i] = 1e-10;
                }

            }else{
                prob[i] = 1e-10;
            }
        }
        out = sum(log(prob));
        return out;
    }
}

data {
    int<lower=1> N;									// number of data items
    vector[2] p[N];                                 // Semantic Word Probabilty p[n][1]:word probability p[n][2]:non-word probability
    int<lower=0> frequency[N];                      // zipf values (representing frequency)
    int<lower=0,upper=1> response[N];				// 1-> word, 0->nonword
    real<lower=0> rt[N];							// rt

    real minRT[N];                                  // minimum RT for each subject of the observed data
    real RTbound;                                   // lower bound or RT across all subjects (e.g., 0.1 second)

    vector[2] k_priors;
	vector[2] sp_trial_var_priors;
    vector[2] g_priors;
    vector[2] m_priors;
    vector[2] alpha_priors;
    vector[2] theta_priors;
    vector[2] b_priors;
    vector[2] drift_variability_priors;
}

transformed data {
	matrix [N, 2] RT;

	for (n in 1:N){
	   RT[n, 1] = rt[n];
	   RT[n, 2] = response[n];
	}
}

parameters {
    real k_word;
    real k_nonword;
    real sp_trial_var_word;
    real sp_trial_var_nonword;
    real g;                              
    real m; 
    real alpha;
    real theta_1;
    real theta_2;
    real b;
    real drift_variability;
}

transformed parameters {
    vector<lower=0> [N] ndt_t;                                      // trial-by-trial ndt
    vector<lower=0> [N] k_t_word;				                    // trial-by-trial
    vector<lower=0> [N] k_t_nonword;				                // trial-by-trial
	vector<lower=0> [N] sp_trial_var_t_word;						// trial-by-trial
	vector<lower=0> [N] sp_trial_var_t_nonword;						// trial-by-trial
	
    vector<lower=0> [N] drift_word_t;				                // trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_nonword_t;				            // trial-by-trial drift rate for predictions
    vector<lower=0> [N] drift_variability_t;

    real<lower=0> transf_k_word;
    real<lower=0> transf_k_nonword;
    real<lower=0> transf_sp_trial_var_word;
    real<lower=0> transf_sp_trial_var_nonword;
    real<lower=0> transf_g;
    real<lower=0> transf_m;
    real<lower=0> transf_alpha;
    real<lower=0> transf_theta_1;
    real<lower=0> transf_theta_2;
    real<lower=0> transf_b;
    real<lower=0> transf_drift_variability;

    transf_k_word = log(1 + exp(k_word));
    transf_k_nonword = log(1 + exp(k_nonword));
	transf_sp_trial_var_word = log(1 + exp(sp_trial_var_word));
	transf_sp_trial_var_nonword = log(1 + exp(sp_trial_var_nonword));
    transf_g = log(1 + exp(g));
    transf_m = log(1 + exp(m));
    transf_alpha = log(1 + exp(alpha));
    transf_theta_1 = log(1 + exp(theta_1));
    transf_theta_2 = log(1 + exp(theta_2));
    transf_b = log(1 + exp(b));
    transf_drift_variability = log(1 + exp(drift_variability));

	for (n in 1:N) {
        k_t_word[n] = transf_k_word;
        k_t_nonword[n] = transf_k_nonword;
		sp_trial_var_t_word[n] = transf_sp_trial_var_word;
		sp_trial_var_t_nonword[n] = transf_sp_trial_var_nonword;
        ndt_t[n] = (transf_m +  transf_g * exp(-frequency[n])) * (minRT[n] - RTbound) + RTbound;
        drift_word_t[n] = (transf_theta_1 + transf_b * frequency[n]) / (1 + exp(-transf_alpha * (p[n][1]-0.5)));
        drift_nonword_t[n] = transf_theta_2 / (1 + exp(-transf_alpha * (p[n][2]-0.5)));
        drift_variability_t[n] = transf_drift_variability;
	}
}

model {
    k_word ~ normal(k_priors[1], k_priors[2]);
    k_nonword ~ normal(k_priors[1], k_priors[2]);
    sp_trial_var_word ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);
    sp_trial_var_nonword ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);
    g ~ normal(g_priors[1], g_priors[2]);
    m ~ normal(m_priors[1], m_priors[2]);
    alpha ~ normal(alpha_priors[1], alpha_priors[2]);
    theta_1 ~ normal(theta_priors[1], theta_priors[2]);
    theta_2 ~ normal(theta_priors[1], theta_priors[2]);
    b ~ normal(b_priors[1], b_priors[2]);
    drift_variability ~ normal(drift_variability_priors[1], drift_variability_priors[2]);

    RT ~ lba(k_t_word, sp_trial_var_t_word, k_t_nonword, sp_trial_var_t_nonword,
             drift_word_t, drift_nonword_t, ndt_t, drift_variability_t);
}

generated quantities {
    vector[N] log_lik;
     {
        for (n in 1:N){
            log_lik[n] = lba_lpdf(block(RT, n, 1, 1, 2)| segment(k_t_word, n, 1), segment(sp_trial_var_t_word, n, 1), segment(k_t_nonword, n, 1),
                                  segment(sp_trial_var_t_nonword, n, 1), segment(drift_word_t, n, 1), segment(drift_nonword_t, n, 1),
                                  segment(ndt_t, n, 1), segment(drift_variability_t, n, 1));
        }
     }
}