// LBA with different threshold and starting point for words and nonwords and
// different drifts across conditions (HF, LF, NW)

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

    // word_threshold (b_word) = A_word + k_word 
    // nonword_threshold (b_nonword) = A_nonword + k_nonword
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
    int<lower=1, upper=3> frequencyCondition[N];    // HF, LF OR NW

    int<lower=0,upper=1> response[N];				// 1-> word, 0->nonword
    real<lower=0> rt[N];							// rt

    real minRT[N];                                  // minimum RT for each subject of the observed data
    real RTbound;                                   // lower bound or RT across all subjects (e.g., 0.1 second)
 
    vector[4] k_priors;
	vector[4] sp_trial_var_priors;
    vector[4] ndt_priors;
	vector[4] drift_priors;
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
    real mu_ndt;
    vector[3] mu_drift_word;
    vector[3] mu_drift_nonword;
    vector[3] mu_drift_variability;

    real<lower=0> sd_k_word;
    real<lower=0> sd_k_nonword;
    real<lower=0> sd_sp_trial_var_word;
    real<lower=0> sd_sp_trial_var_nonword;
    real<lower=0> sd_ndt;
    vector<lower=0>[3] sd_drift_word;
    vector<lower=0>[3] sd_drift_nonword;
    vector<lower=0>[3] sd_drift_variability;

    real z_k_word[L];
    real z_k_nonword[L];
    real z_sp_trial_var_word[L];
    real z_sp_trial_var_nonword[L];
    real z_ndt[L];
    vector[3] z_drift_word[L];
    vector[3] z_drift_nonword[L];
    vector[3] z_drift_variability[L];
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
	real<lower=0> ndt_sbj[L];
    vector<lower=0>[3] drift_word_sbj[L];
	vector<lower=0>[3] drift_nonword_sbj[L];
    vector<lower=0>[3] drift_variability_sbj[L];

    real<lower=0> transf_mu_k_word;
    real<lower=0> transf_mu_k_nonword;
    real<lower=0> transf_mu_sp_trial_var_word;
    real<lower=0> transf_mu_sp_trial_var_nonword;
    real<lower=0> transf_mu_ndt;
	vector<lower=0>[3] transf_mu_drift_word;
	vector<lower=0>[3] transf_mu_drift_nonword;
    vector<lower=0>[3] transf_mu_drift_variability;

    transf_mu_k_word = log(1 + exp(mu_k_word));
    transf_mu_k_nonword = log(1 + exp(mu_k_nonword));
	transf_mu_sp_trial_var_word = log(1 + exp(mu_sp_trial_var_word));
	transf_mu_sp_trial_var_nonword = log(1 + exp(mu_sp_trial_var_nonword));
	transf_mu_ndt = log(1 + exp(mu_ndt));

    for(i in 1:3)
    {
	    transf_mu_drift_word[i] = log(1 + exp(mu_drift_word[i]));
	    transf_mu_drift_nonword[i] = log(1 + exp(mu_drift_nonword[i]));
        transf_mu_drift_variability[i] = log(1 + exp(mu_drift_variability[i]));
    }

     for (l in 1:L){
        k_sbj_word[l] = log(1 + exp(mu_k_word + z_k_word[l] * sd_k_word));
        k_sbj_nonword[l] = log(1 + exp(mu_k_nonword + z_k_nonword[l] * sd_k_nonword));
		sp_trial_var_sbj_word[l] = log(1 + exp(mu_sp_trial_var_word + z_sp_trial_var_word[l] * sd_sp_trial_var_word));
		sp_trial_var_sbj_nonword[l] = log(1 + exp(mu_sp_trial_var_nonword + z_sp_trial_var_nonword[l] * sd_sp_trial_var_nonword));
        ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l] * sd_ndt));
        for(i in 1:3)
        {
	        drift_word_sbj[l][i] = log(1 + exp(mu_drift_word[i] + z_drift_word[l][i] * sd_drift_word[i]));
		    drift_nonword_sbj[l][i] = log(1 + exp(mu_drift_nonword[i] + z_drift_nonword[l][i] * sd_drift_nonword[i]));
            drift_variability_sbj[l][i] = log(1 + exp(mu_drift_variability[i] + z_drift_variability[l][i] * sd_drift_variability[i]));
        }
	}

	for (n in 1:N) {
        k_t_word[n] = k_sbj_word[participant[n]];
        k_t_nonword[n] = k_sbj_nonword[participant[n]];
		sp_trial_var_t_word[n] = sp_trial_var_sbj_word[participant[n]];
		sp_trial_var_t_nonword[n] = sp_trial_var_sbj_nonword[participant[n]];
        ndt_t[n] = ndt_sbj[participant[n]] * (minRT[n] - RTbound) + RTbound;
		drift_word_t[n] = drift_word_sbj[participant[n]][frequencyCondition[n]];
		drift_nonword_t[n] = drift_nonword_sbj[participant[n]][frequencyCondition[n]];
        drift_variability_t[n] = drift_variability_sbj[participant[n]][frequencyCondition[n]];
	}
}

model {
    mu_k_word ~ normal(k_priors[1], k_priors[2]);
    mu_k_nonword ~ normal(k_priors[1], k_priors[2]);
    mu_sp_trial_var_word ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);
    mu_sp_trial_var_nonword ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);
    mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);
    mu_drift_word ~ normal(drift_priors[1], drift_priors[2]);
   	mu_drift_nonword ~ normal(drift_priors[1], drift_priors[2]);
    mu_drift_variability ~ normal(drift_variability_priors[1], drift_variability_priors[2]);


    sd_k_word ~ normal(k_priors[3], k_priors[4]);
    sd_k_nonword ~ normal(k_priors[3], k_priors[4]);
    sd_sp_trial_var_word ~ normal(sp_trial_var_priors[3], sp_trial_var_priors[4]);
    sd_sp_trial_var_nonword ~ normal(sp_trial_var_priors[3], sp_trial_var_priors[4]);
    sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);
    sd_drift_word ~ normal(drift_priors[3], drift_priors[4]);
   	sd_drift_nonword ~ normal(drift_priors[3], drift_priors[4]);
    sd_drift_variability ~ normal(drift_variability_priors[3], drift_variability_priors[4]);

    z_k_word ~ normal(0, 1);
    z_k_nonword ~ normal(0, 1);
    z_sp_trial_var_word ~ normal(0, 1);
    z_sp_trial_var_nonword ~ normal(0, 1);
    z_ndt ~ normal(0, 1);
    for (l in 1:L) {
        z_drift_word[l] ~ normal(0, 1);
   	    z_drift_nonword[l] ~ normal(0, 1);
        z_drift_variability[l] ~ normal(0, 1);
    }

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