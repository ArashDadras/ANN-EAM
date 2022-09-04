// LBA with same threshold and starting point for words and nonwords but
// with same drifts across conditions

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
    
     // threshold (b) = A + k 
     // tau = ndt 
     real lba_lpdf(matrix RT, vector k, vector A, vector drift_word,
                   vector drift_nonword, vector tau){

          real t;
          real b;
          real cdf;
          real pdf;
          vector[rows(RT)] prob;
          real out;
          real prob_neg;
          real s;
          s = 1;

          for (i in 1:rows(RT)){
               b = A[i] + k[i];
               t = RT[i,1] - tau[i];
               if(t > 0){
                    cdf = 1;

                    if(RT[i,2] == 1){
                      pdf = lba_pdf(t, b, A[i], drift_word[i], s);
                      cdf = 1-lba_cdf(t, b, A[i], drift_nonword[i], s);
                    }
                    else{
                      pdf = lba_pdf(t, b, A[i], drift_nonword[i], s);
                      cdf = 1-lba_cdf(t, b, A[i], drift_word[i], s);
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

    int<lower=0,upper=1> response[N];				// 1-> word, 0->nonword
    real<lower=0> rt[N];							// rt

    real minRT[N];                                  // minimum RT for each subject of the observed data
    real RTbound;                                   // lower bound or RT across all subjects (e.g., 0.1 second)

    vector[4] k_priors;
    vector[4] A_priors;
    vector[4] ndt_priors;
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
    real mu_k;
    real mu_A;
    real mu_ndt;
    real mu_drift_word;
    real mu_drift_nonword;

    real<lower=0> sd_k;
	real<lower=0> sd_A;
    real<lower=0> sd_ndt;
	real<lower=0> sd_drift_word;
	real<lower=0> sd_drift_nonword;

    real z_k[L];
    real z_A[L];
    real z_ndt[L];
    real z_drift_word[L];
    real z_drift_nonword[L];
}

transformed parameters {
    vector<lower=0> [N] k_t;   		                // trial-by-trial
	vector<lower=0> [N] A_t;					    // trial-by-trial
    vector<lower=0> [N] ndt_t;				        // trial-by-trial ndt
	vector<lower=0> [N] drift_word_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_nonword_t;		    // trial-by-trial drift rate for predictions

    real<lower=0> k_sbj[L];
	real<lower=0> A_sbj[L];
	real<lower=0> ndt_sbj[L];
    real<lower=0> drift_word_sbj[L];
	real<lower=0> drift_nonword_sbj[L];

    real<lower=0> transf_mu_k;
    real<lower=0> transf_mu_A;
    real<lower=0> transf_mu_ndt;
	real<lower=0> transf_mu_drift_word;
	real<lower=0> transf_mu_drift_nonword;

    transf_mu_k = log(1 + exp(mu_k));
	transf_mu_A = log(1 + exp(mu_A));
	transf_mu_ndt = log(1 + exp(mu_ndt));
	transf_mu_drift_word = log(1 + exp(mu_drift_word));
	transf_mu_drift_nonword = log(1 + exp(mu_drift_nonword));

    for (l in 1:L) {
        k_sbj[l] = log(1 + exp(mu_k + z_k[l] * sd_k));
		A_sbj[l] = log(1 + exp(mu_A + z_A[l] * sd_A));
        ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l] * sd_ndt));
		drift_word_sbj[l] = log(1 + exp(mu_drift_word + z_drift_word[l] * sd_drift_word));
		drift_nonword_sbj[l] = log(1 + exp(mu_drift_nonword + z_drift_nonword[l] * sd_drift_nonword));
	}

	for (n in 1:N) {
        k_t[n] = k_sbj[participant[n]];
		A_t[n] = A_sbj[participant[n]];
        ndt_t[n] = ndt_sbj[participant[n]] * (minRT[n] - RTbound) + RTbound;
		drift_word_t[n] = drift_word_sbj[participant[n]];
		drift_nonword_t[n] = drift_nonword_sbj[participant[n]];
	}
}

model {
     mu_k ~ normal(k_priors[1], k_priors[2]);
     mu_A ~ normal(A_priors[1], A_priors[2]);
     mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);
     mu_drift_word ~ normal(drift_priors[1], drift_priors[2]);
   	 mu_drift_nonword ~ normal(drift_priors[1], drift_priors[2]);


     sd_k ~ normal(k_priors[3], k_priors[4]);
     sd_A ~ normal(A_priors[3], A_priors[4]);
     sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);
     sd_drift_word ~ normal(drift_priors[3], drift_priors[4]);
   	 sd_drift_nonword ~ normal(drift_priors[3], drift_priors[4]);

     z_k ~ normal(0, 1);
     z_A ~ normal(0, 1);
     z_ndt ~ normal(0, 1);
     z_drift_word ~ normal(0, 1);
   	 z_drift_nonword ~ normal(0, 1);

     RT ~ lba(k_t, A_t, drift_word_t, drift_nonword_t, ndt_t);
}

generated quantities {
    vector[N] log_lik;
  	{
    	for (n in 1:N){
    		log_lik[n] = lba_lpdf(block(RT, n, 1, 1, 2)| segment(k_t, n, 1), segment(A_t, n, 1),
                                  segment(drift_word_t, n, 1), segment(drift_nonword_t, n, 1),
                                  segment(ndt_t, n, 1));
    	}
  	}
}