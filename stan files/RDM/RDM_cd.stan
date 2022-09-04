// RDM with different threshold for words and nonwords and
// different drifts across conditions

functions {
    real race_pdf(real t, real b, real v){
        real pdf;
        pdf = b/sqrt(2 * pi() * pow(t, 3)) * exp(-pow(v*t-b, 2) / (2*t));
        return pdf;
    }

    real race_cdf(real t, real b, real v){
        real cdf;
        cdf = Phi((v*t-b)/sqrt(t)) + exp(2*v*b) * Phi(-(v*t+b)/sqrt(t));
        return cdf;
    }

    real race_lpdf(matrix RT, vector ndt, vector b_word, vector b_nonword, vector drift_word, vector drift_nonword){

        real t;
        vector[rows(RT)] prob;
        real cdf;
        real pdf;
        real out;

        for (i in 1:rows(RT)){
            t = RT[i,1] - ndt[i];
            if(t > 0){
                if(RT[i,2] == 1){
                    pdf = race_pdf(t, b_word[i], drift_word[i]);
                    cdf = 1 - race_cdf(t, b_nonword[i], drift_nonword[i]);
                }
                else{
                    pdf = race_pdf(t, b_nonword[i], drift_nonword[i]);
                    cdf = 1 - race_cdf(t, b_word[i], drift_word[i]);
                }
                prob[i] = pdf*cdf;

                if(prob[i] < 1e-10){
                    prob[i] = 1e-10;
                }
            }
            else{
                prob[i] = 1e-10;
            }
        }
        out = sum(log(prob));
        return out;
    }
}

data {
    int<lower=1> N;                                 // number of data items
    int<lower=1> L;                                 // number of levels
    int<lower=1, upper=L> participant[N];           // level (participant)
    int<lower=1, upper=3> frequencyCondition[N];    // HF, LF OR NW
    int<lower=0,upper=1> response[N];               // 1-> word, 0->nonword
    real<lower=0> rt[N];                            // rt
    
    real minRT[N];                                  // minimum RT for each subject of the observed data
    real RTbound;                                   // lower bound or RT across all subjects (e.g., 0.1 second)
                         
    vector[4] drift_priors;                         // mean and sd of the group mean and of the group sd hyper-priors
    vector[4] threshold_priors;                     // mean and sd of the group mean and of the group sd hyper-priors
    vector[4] ndt_priors;                           // mean and sd of the group mean and of the group sd hyper-priors
}

transformed data {
    matrix [N, 2] RT;

    for (n in 1:N)
    {
        RT[n, 1] = rt[n];
        RT[n, 2] = response[n];
    }
}

parameters {
    
    real mu_ndt;
    real mu_threshold_word;
    real mu_threshold_nonword;
    vector[3] mu_drift_word;                           // 3 drift for 3 conditions: 1=HF, 2=LF, 3=NW
    vector[3] mu_drift_nonword;                        // 3 drift for 3 conditions: 1=HF, 2=LF, 3=NW

    real<lower=0> sd_ndt;
    real<lower=0> sd_threshold_word;
    real<lower=0> sd_threshold_nonword;
    real<lower=0> sd_drift_word[3];
    real<lower=0> sd_drift_nonword[3];

    real z_ndt[L];
    real z_threshold_word[L];
    real z_threshold_nonword[L];
    vector[3] z_drift_word[L];
    vector[3] z_drift_nonword[L];
    
}

transformed parameters {
    vector<lower=0>[N] drift_word_t;                     // trial-by-trial drift rate for predictions
    vector<lower=0>[N] drift_nonword_t;                  // trial-by-trial drift rate for predictions
    vector<lower=0>[N] threshold_t_word;                 // trial-by-trial word threshold
    vector<lower=0>[N] threshold_t_nonword;              // trial-by-trial nonword threshold
    vector<lower=0>[N] ndt_t;                            // trial-by-trial ndt

    vector<lower=0>[3] drift_word_sbj[L];
    vector<lower=0>[3] drift_nonword_sbj[L];
    real<lower=0> threshold_word_sbj[L];
    real<lower=0> threshold_nonword_sbj[L];
    real<lower=0> ndt_sbj[L];

    vector<lower=0>[3] transf_mu_drift_word;
    vector<lower=0>[3] transf_mu_drift_nonword;
    real<lower=0> transf_mu_threshold_word;
    real<lower=0> transf_mu_threshold_nonword;
    real<lower=0> transf_mu_ndt;
    
    for(i in 1:3)
    {
        transf_mu_drift_word[i] = log(1 + exp(mu_drift_word[i]));
        transf_mu_drift_nonword[i] = log(1 + exp(mu_drift_nonword[i]));
    }
    transf_mu_threshold_word = log(1 + exp(mu_threshold_word));
    transf_mu_threshold_nonword = log(1 + exp(mu_threshold_nonword));
    transf_mu_ndt = log(1 + exp(mu_ndt));

    for (l in 1:L) {
        for(i in 1:3)
        {
            drift_word_sbj[l][i] = log(1 + exp(mu_drift_word[i] + z_drift_word[l][i] * sd_drift_word[i]));
            drift_nonword_sbj[l][i] = log(1 + exp(mu_drift_nonword[i] + z_drift_nonword[l][i] * sd_drift_nonword[i]));
        }
        threshold_word_sbj[l] = log(1 + exp(mu_threshold_word + z_threshold_word[l] * sd_threshold_word));
        threshold_nonword_sbj[l] = log(1 + exp(mu_threshold_nonword + z_threshold_nonword[l] * sd_threshold_nonword));
        ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l] * sd_ndt));
    }

    for (n in 1:N) {
        drift_word_t[n] = drift_word_sbj[participant[n]][frequencyCondition[n]];
        drift_nonword_t[n] = drift_nonword_sbj[participant[n]][frequencyCondition[n]];
        threshold_t_word[n] = threshold_word_sbj[participant[n]];
        threshold_t_nonword[n] = threshold_nonword_sbj[participant[n]];
        ndt_t[n] = ndt_sbj[participant[n]] * (minRT[n] - RTbound) + RTbound;
    }
}

model {
    mu_drift_word ~ normal(drift_priors[1], drift_priors[2]);
    mu_drift_nonword ~ normal(drift_priors[1], drift_priors[2]);
    mu_threshold_word ~ normal(threshold_priors[1], threshold_priors[2]);
    mu_threshold_nonword ~ normal(threshold_priors[1], threshold_priors[2]);
    mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);

    sd_drift_word ~ normal(drift_priors[3], drift_priors[4]);
    sd_drift_nonword ~ normal(drift_priors[3], drift_priors[4]);
    sd_threshold_word ~ normal(threshold_priors[3], threshold_priors[4]);
    sd_threshold_nonword ~ normal(threshold_priors[3], threshold_priors[4]);
    sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);

    for (l in 1:L) {
        z_drift_word[l] ~ normal(0, 1);
        z_drift_nonword[l] ~ normal(0, 1);
    }
    z_threshold_word ~ normal(0, 1);
    z_threshold_nonword ~ normal(0, 1);
    z_ndt ~ normal(0, 1);

    RT ~ race(ndt_t, threshold_t_word, threshold_t_nonword, drift_word_t, drift_nonword_t);
}

generated quantities {
    vector[N] log_lik;
    {
    for (n in 1:N){
        log_lik[n] = race_lpdf(block(RT, n, 1, 1, 2)| segment(ndt_t, n, 1), segment(threshold_t_word, n, 1), segment(threshold_t_nonword, n, 1), segment(drift_word_t, n, 1), segment(drift_nonword_t, n, 1));
    }
    }
}