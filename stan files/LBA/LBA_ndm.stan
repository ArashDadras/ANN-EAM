// LBA with different threshold and starting point for words and nonwords and
// different drifts across conditions (HF, LF, NW)
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

    
     // word_threshold (b_word) = A_word + k_word 
     // nonword_threshold (b_nonword) = A_nonword + k_nonword
     // tau = ndt 
     real lba_lpdf(matrix RT, vector k_word, vector A_word, vector k_nonword, vector A_nonword,
                   vector drift_word, vector drift_nonword, vector tau){

          real t;
          real b_word;
          real b_nonword;
          real cdf;
          real pdf;
          vector[rows(RT)] prob;
          real out;
          real prob_neg;
          real s;
          s = 1;

          for (i in 1:rows(RT)){
               b_word = A_word[i] + k_word[i];
               b_nonword = A_nonword[i] + k_nonword[i];
               t = RT[i,1] - tau[i];
               if(t > 0){
                    cdf = 1;

                    if(RT[i,2] == 1){
                      pdf = lba_pdf(t, b_word, A_word[i], drift_word[i], s);
                      cdf = 1-lba_cdf(t, b_nonword, A_nonword[i], drift_nonword[i], s);
                    }
                    else{
                      pdf = lba_pdf(t, b_nonword, A_nonword[i], drift_nonword[i], s);
                      cdf = 1-lba_cdf(t, b_word, A_word[i], drift_word[i], s);
                    }
                    prob_neg = Phi(-drift_word[i]/s) * Phi(-drift_nonword[i]/s);
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
    int<lower=0> frequency[N];                      // zipf values (representing frequency)
    int<lower=1, upper=3> frequencyCondition[N];    // HF, LF OR NW

    int<lower=0,upper=1> response[N];				// 1-> word, 0->nonword
    real<lower=0> rt[N];							// rt

    real minRT[N];                                  // minimum RT for each subject of the observed data
    real RTbound;                                   // lower bound or RT across all subjects (e.g., 0.1 second)
    
    vector[4] k_priors;
    vector[4] A_priors;
    vector[4] g_priors;
    vector[4] m_priors;
    vector[4] drift_priors;
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
    real mu_A_word;
    real mu_k_nonword;
    real mu_A_nonword;
    real mu_g;                              
    real mu_m; 
    vector[3] mu_drift_word;
    vector[3] mu_drift_nonword;

    real<lower=0> sd_k_word;
	real<lower=0> sd_A_word;
    real<lower=0> sd_k_nonword;
	real<lower=0> sd_A_nonword;
    real<lower=0> sd_g;
    real<lower=0> sd_m;
	vector<lower=0>[3] sd_drift_word;
	vector<lower=0>[3] sd_drift_nonword;

    real z_k_word[L];
    real z_A_word[L];
    real z_k_nonword[L];
    real z_A_nonword[L];
    real z_g[L];
    real z_m[L];
    vector[3] z_drift_word[L];
    vector[3] z_drift_nonword[L];
}

transformed parameters {
    vector<lower=0> [N] k_t_word;   		        // trial-by-trial
	vector<lower=0> [N] A_t_word;					// trial-by-trial
    vector<lower=0> [N] k_t_nonword;   		        // trial-by-trial
	vector<lower=0> [N] A_t_nonword;				// trial-by-trial
    vector<lower=0> [N] ndt_t;				        // trial-by-trial ndt
	vector<lower=0> [N] drift_word_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_nonword_t;		    // trial-by-trial drift rate for predictions

    real<lower=0> k_word_sbj[L];
	real<lower=0> A_word_sbj[L];
    real<lower=0> k_nonword_sbj[L];
	real<lower=0> A_nonword_sbj[L];
    real g_sbj[L];
    real m_sbj[L];
    vector<lower=0>[3] drift_word_sbj[L];
	vector<lower=0>[3] drift_nonword_sbj[L];

    real<lower=0> transf_mu_k_word;
    real<lower=0> transf_mu_A_word;
    real<lower=0> transf_mu_k_nonword;
    real<lower=0> transf_mu_A_nonword;
    real<lower=0> transf_mu_g;
    real<lower=0> transf_mu_m;
	vector<lower=0>[3] transf_mu_drift_word;
	vector<lower=0>[3] transf_mu_drift_nonword;

    transf_mu_k_word = log(1 + exp(mu_k_word));
	transf_mu_A_word = log(1 + exp(mu_A_word));
    transf_mu_k_nonword = log(1 + exp(mu_k_nonword));
	transf_mu_A_nonword = log(1 + exp(mu_A_nonword));
    transf_mu_g = log(1 + exp(mu_g));
    transf_mu_m = log(1 + exp(mu_m));

    for(i in 1:3)
    {
        transf_mu_drift_word[i] = log(1 + exp(mu_drift_word[i]));
	    transf_mu_drift_nonword[i] = log(1 + exp(mu_drift_nonword[i]));
    }

    for (l in 1:L) {
        k_word_sbj[l] = log(1 + exp(mu_k_word + z_k_word[l] * sd_k_word));
		A_word_sbj[l] = log(1 + exp(mu_A_word + z_A_word[l] * sd_A_word));
        k_nonword_sbj[l] = log(1 + exp(mu_k_nonword + z_k_nonword[l] * sd_k_nonword));
		A_nonword_sbj[l] = log(1 + exp(mu_A_nonword + z_A_nonword[l] * sd_A_nonword));
        g_sbj[l] = log(1 + exp(mu_g + z_g[l] * sd_g));
        m_sbj[l] = log(1 + exp(mu_m + z_m[l] * sd_m));
        for(i in 1:3)
        {
		    drift_word_sbj[l][i] = log(1 + exp(mu_drift_word[i] + z_drift_word[l][i] * sd_drift_word[i]));
		    drift_nonword_sbj[l][i] = log(1 + exp(mu_drift_nonword[i] + z_drift_nonword[l][i] * sd_drift_nonword[i]));
        }
	}

	for (n in 1:N) {
        k_t_word[n] = k_word_sbj[participant[n]];
		A_t_word[n] = A_word_sbj[participant[n]];
        k_t_nonword[n] = k_nonword_sbj[participant[n]];
		A_t_nonword[n] = A_nonword_sbj[participant[n]];
        ndt_t[n] = (m_sbj[participant[n]] +  g_sbj[participant[n]] * exp(-frequency[n])) * (minRT[n] - RTbound) + RTbound;
		drift_word_t[n] = drift_word_sbj[participant[n]][frequencyCondition[n]];
		drift_nonword_t[n] = drift_nonword_sbj[participant[n]][frequencyCondition[n]];
	}
}

model {
    mu_k_word ~ normal(k_priors[1], k_priors[2]);
    mu_A_word ~ normal(A_priors[1], A_priors[2]);
    mu_k_nonword ~ normal(k_priors[1], k_priors[2]);
    mu_A_nonword ~ normal(A_priors[1], A_priors[2]);
    mu_g ~ normal(g_priors[1], g_priors[2]);
    mu_m ~ normal(m_priors[1], m_priors[2]);
    mu_drift_word ~ normal(drift_priors[1], drift_priors[2]);
   	mu_drift_nonword ~ normal(drift_priors[1], drift_priors[2]);

    sd_k_word ~ normal(k_priors[3], k_priors[4]);
    sd_A_word ~ normal(A_priors[3], A_priors[4]);
    sd_k_nonword ~ normal(k_priors[3], k_priors[4]);
    sd_A_nonword ~ normal(A_priors[3], A_priors[4]);
    sd_g ~ normal(g_priors[3], g_priors[4]);
    sd_m ~ normal(m_priors[3], m_priors[4]); 
    sd_drift_word ~ normal(drift_priors[3], drift_priors[4]);
   	sd_drift_nonword ~ normal(drift_priors[3], drift_priors[4]);

    z_k_word ~ normal(0, 1);
    z_A_word ~ normal(0, 1);
    z_k_nonword ~ normal(0, 1);
    z_A_nonword ~ normal(0, 1);
    z_g ~ normal(0, 1);
    z_m ~ normal(0, 1);
    for (l in 1:L) {
        z_drift_word[l] ~ normal(0, 1);
        z_drift_nonword[l] ~ normal(0, 1);
    }

    RT ~ lba(k_t_word, A_t_word, k_t_nonword, A_t_nonword,
             drift_word_t, drift_nonword_t, ndt_t);
}

generated quantities {
    vector[N] log_lik;
  	{
    	for (n in 1:N){
    		log_lik[n] = lba_lpdf(block(RT, n, 1, 1, 2)| segment(k_t_word, n, 1), segment(A_t_word, n, 1),
                                  segment(k_t_nonword, n, 1), segment(A_t_nonword, n, 1),
                                  segment(drift_word_t, n, 1), segment(drift_nonword_t, n, 1),
                                  segment(ndt_t, n, 1));
    	}
  	}
}