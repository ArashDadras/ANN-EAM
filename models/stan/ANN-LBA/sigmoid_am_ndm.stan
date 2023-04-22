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
    int<lower=1> L;									// number of levels
    int<lower=1, upper=L> participant[N];			// level (participant)
    vector[2] p[N];                                 // Semantic Word Probabilty p[n][1]:word probability p[n][2]:non-word probability
    int<lower=0> frequency[N];                      // zipf values (representing frequency)
    int<lower=0,upper=1> response[N];				// 1-> word, 0->nonword
    real<lower=0> rt[N];							// rt

    real minRT[N];                                  // minimum RT for each subject of the observed data
    real RTbound;                                   // lower bound or RT across all subjects (e.g., 0.1 second)

    vector[4] k_priors;
	vector[4] sp_trial_var_priors;
    vector[4] g_priors;
    vector[4] m_priors;
    vector[4] alpha_priors;
    vector[4] theta_priors;
    vector[4] b_priors;
    vector[4] drift_variability_priors;
}

transformed data {
	matrix [N, 2] RT;

	for (n in 1:N){
	   RT[n, 1] = rt[n];
	   RT[n, 2] = response[n];
	}
}

parameters {
    real mu_k_word;
    real mu_k_nonword;
    real mu_sp_trial_var_word;
    real mu_sp_trial_var_nonword;
    real mu_g;                              
    real mu_m; 
    real mu_alpha;
    real mu_theta_1;
    real mu_theta_2;
    real mu_b;
    real mu_drift_variability;

    real<lower=0> sd_k_word;
    real<lower=0> sd_k_nonword;
    real<lower=0> sd_sp_trial_var_word;
    real<lower=0> sd_sp_trial_var_nonword;
    real<lower=0> sd_g;
    real<lower=0> sd_m;
    real<lower=0> sd_alpha;
    real<lower=0> sd_theta_1;
    real<lower=0> sd_theta_2;
    real<lower=0> sd_b;
    real<lower=0> sd_drift_variability;

    real z_k_word[L];
    real z_k_nonword[L];
    real z_sp_trial_var_word[L];
    real z_sp_trial_var_nonword[L];
    real z_g[L];
    real z_m[L];
    real z_alpha[L];
    real z_theta_1[L];
    real z_theta_2[L];
    real z_b[L];
    real z_drift_variability[L];
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

    real<lower=0> k_sbj_word[L];
    real<lower=0> k_sbj_nonword[L];
	real<lower=0> sp_trial_var_sbj_word[L];
	real<lower=0> sp_trial_var_sbj_nonword[L];
    real g_sbj[L];
    real m_sbj[L];
    real<lower=0> alpha_sbj[L];
    real<lower=0> theta_1_sbj[L];
    real<lower=0> theta_2_sbj[L];
    real<lower=0> b_sbj[L];
    real<lower=0> drift_variability_sbj[L];

    real<lower=0> transf_mu_k_word;
    real<lower=0> transf_mu_k_nonword;
    real<lower=0> transf_mu_sp_trial_var_word;
    real<lower=0> transf_mu_sp_trial_var_nonword;
    real<lower=0> transf_mu_g;
    real<lower=0> transf_mu_m;
    real<lower=0> transf_mu_alpha;
    real<lower=0> transf_mu_theta_1;
    real<lower=0> transf_mu_theta_2;
    real<lower=0> transf_mu_b;
    real<lower=0> transf_mu_drift_variability;

    transf_mu_k_word = log(1 + exp(mu_k_word));
    transf_mu_k_nonword = log(1 + exp(mu_k_nonword));
	transf_mu_sp_trial_var_word = log(1 + exp(mu_sp_trial_var_word));
	transf_mu_sp_trial_var_nonword = log(1 + exp(mu_sp_trial_var_nonword));
    transf_mu_g = log(1 + exp(mu_g));
    transf_mu_m = log(1 + exp(mu_m));
    transf_mu_alpha = log(1 + exp(mu_alpha));
    transf_mu_theta_1 = log(1 + exp(mu_theta_1));
    transf_mu_theta_2 = log(1 + exp(mu_theta_2));
    transf_mu_b = log(1 + exp(mu_b));
    transf_mu_drift_variability = log(1 + exp(mu_drift_variability));

    for (l in 1:L){
        k_sbj_word[l] = log(1 + exp(mu_k_word + z_k_word[l] * sd_k_word));
        k_sbj_nonword[l] = log(1 + exp(mu_k_nonword + z_k_nonword[l] * sd_k_nonword));
		sp_trial_var_sbj_word[l] = log(1 + exp(mu_sp_trial_var_word + z_sp_trial_var_word[l] * sd_sp_trial_var_word));
		sp_trial_var_sbj_nonword[l] = log(1 + exp(mu_sp_trial_var_nonword + z_sp_trial_var_nonword[l] * sd_sp_trial_var_nonword));
        g_sbj[l] = log(1 + exp(mu_g + z_g[l] * sd_g));
        m_sbj[l] = log(1 + exp(mu_m + z_m[l] * sd_m));
        alpha_sbj[l] = log(1 + exp(mu_alpha + z_alpha[l] * sd_alpha));
        theta_1_sbj[l] = log(1 + exp(mu_theta_1 + z_theta_1[l] * sd_theta_1));
        theta_2_sbj[l] = log(1 + exp(mu_theta_2 + z_theta_2[l] * sd_theta_2));
        b_sbj[l] = log(1 + exp(mu_b + z_b[l] * sd_b));
        drift_variability_sbj[l] = log(1 + exp(mu_drift_variability + z_drift_variability[l] * sd_drift_variability));
	}

	for (n in 1:N) {
        k_t_word[n] = k_sbj_word[participant[n]];
        k_t_nonword[n] = k_sbj_nonword[participant[n]];
		sp_trial_var_t_word[n] = sp_trial_var_sbj_word[participant[n]];
		sp_trial_var_t_nonword[n] = sp_trial_var_sbj_nonword[participant[n]];
        ndt_t[n] = (m_sbj[participant[n]] +  g_sbj[participant[n]] * exp(-frequency[n])) * (minRT[n] - RTbound) + RTbound;
        drift_word_t[n] = (theta_1_sbj[participant[n]] + b_sbj[participant[n]] * frequency[n]) / (1 + exp(-alpha_sbj[participant[n]] * (p[n][1]-0.5)));
        drift_nonword_t[n] = theta_2_sbj[participant[n]] / (1 + exp(-alpha_sbj[participant[n]] * (p[n][2]-0.5)));
        drift_variability_t[n] = drift_variability_sbj[participant[n]];
	}
}

model {
    mu_k_word ~ normal(k_priors[1], k_priors[2]);
    mu_k_nonword ~ normal(k_priors[1], k_priors[2]);
    mu_sp_trial_var_word ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);
    mu_sp_trial_var_nonword ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);
    mu_g ~ normal(g_priors[1], g_priors[2]);
    mu_m ~ normal(m_priors[1], m_priors[2]);
    mu_alpha ~ normal(alpha_priors[1], alpha_priors[2]);
    mu_theta_1 ~ normal(theta_priors[1], theta_priors[2]);
    mu_theta_2 ~ normal(theta_priors[1], theta_priors[2]);
    mu_b ~ normal(b_priors[1], b_priors[2]);
    mu_drift_variability ~ normal(drift_variability_priors[1], drift_variability_priors[2]);


    sd_k_word ~ normal(k_priors[3], k_priors[4]);
    sd_k_nonword ~ normal(k_priors[3], k_priors[4]);
    sd_sp_trial_var_word ~ normal(sp_trial_var_priors[3], sp_trial_var_priors[4]);
    sd_sp_trial_var_nonword ~ normal(sp_trial_var_priors[3], sp_trial_var_priors[4]);
    sd_g ~ normal(g_priors[3], g_priors[4]);
    sd_m ~ normal(m_priors[3], m_priors[4]);
    sd_alpha ~ normal(alpha_priors[3], alpha_priors[4]);
    sd_theta_1 ~ normal(theta_priors[3], theta_priors[4]);
    sd_theta_2 ~ normal(theta_priors[3], theta_priors[4]);
    sd_b ~ normal(b_priors[3], b_priors[4]);
    sd_drift_variability ~ normal(drift_variability_priors[3], drift_variability_priors[4]);

    z_k_word ~ normal(0, 1);
    z_k_nonword ~ normal(0, 1);
    z_sp_trial_var_word ~ normal(0, 1);
    z_sp_trial_var_nonword ~ normal(0, 1);
    z_g ~ normal(0, 1);
    z_m ~ normal(0, 1);
    z_alpha ~ normal(0, 1);
    z_theta_1 ~ normal(0, 1);
    z_theta_2 ~ normal(0, 1);
    z_b ~ normal(0, 1);
    z_drift_variability ~ normal(0, 1);

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