// ANN-RDM with sigmoid mapping function for drifts with asymptote modulation,
// different threshold for words and nonwords,
// with non-decision time modulation

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
    vector[2] p[N];                                 // Semantic Word Probabilty p[n][1]:word probability p[n][2]:non-word probability
    int<lower=0> frequency[N];                      // zipf values (representing frequency)
    int<lower=0,upper=1> response[N];               // 1-> word, 0->nonword
    real<lower=0> rt[N];                            // rt
    
    real minRT[N];                                  // minimum RT for each subject of the observed data
    real RTbound;                                   // lower bound or RT across all subjects (e.g., 0.1 second)
                         
    vector[2] threshold_priors;                     
    vector[2] g_priors;
    vector[2] m_priors;
    vector[2] alpha_priors;
    vector[2] b_priors;
    vector[2] k_priors;
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

    real g;                              
    real m; 
    real threshold_word;
    real threshold_nonword;
    real alpha;
    real b;
    real k_1;
    real k_2;
}

transformed parameters {
    vector<lower=0>[N] drift_word_t;                     // trial-by-trial drift rate for predictions
    vector<lower=0>[N] drift_nonword_t;                  // trial-by-trial drift rate for predictions
    vector<lower=0>[N] threshold_t_word;                 // trial-by-trial word threshold
    vector<lower=0>[N] threshold_t_nonword;              // trial-by-trial nonword threshold
    vector [N] ndt_t;                                    // trial-by-trial ndt

    real<lower=0> transf_alpha;
    real<lower=0> transf_b;
    real<lower=0> transf_k_1;
    real<lower=0> transf_k_2;
    real<lower=0> transf_threshold_word;
    real<lower=0> transf_threshold_nonword;
    real<lower=0> transf_g;
    real<lower=0> transf_m;
    
    transf_alpha = log(1 + exp(alpha));
    transf_b = log(1 + exp(b));
    transf_k_1 = log(1 + exp(k_1));
    transf_k_2 = log(1 + exp(k_2));
    transf_threshold_word = log(1 + exp(threshold_word));
    transf_threshold_nonword = log(1 + exp(threshold_nonword));
    transf_g = log(1 + exp(g));
    transf_m = log(1 + exp(m));

    for (n in 1:N) {
        drift_word_t[n] = (transf_k_1 + transf_b * frequency[n]) / (1 + exp(-transf_alpha * (p[n][1]-0.5)));
        drift_nonword_t[n] = transf_k_2 / (1 + exp(-transf_alpha * (p[n][2]-0.5)));
       
        threshold_t_word[n] = transf_threshold_word;
        threshold_t_nonword[n] = transf_threshold_nonword;
        
        ndt_t[n] = (transf_m +  transf_g * exp(-frequency[n])) * (minRT[n] - RTbound) + RTbound;
    }
}

model {
    threshold_word ~ normal(threshold_priors[1], threshold_priors[2]);
    threshold_nonword ~ normal(threshold_priors[1], threshold_priors[2]);
    g ~ normal(g_priors[1], g_priors[2]);
    m ~ normal(m_priors[1], m_priors[2]);
    alpha ~ normal(alpha_priors[1], alpha_priors[2]);
    b ~ normal(b_priors[1], b_priors[2]);
    k_1 ~ normal(k_priors[1], k_priors[2]);
    k_2 ~ normal(k_priors[1], k_priors[2]);
    
    RT ~ race(ndt_t, threshold_t_word, threshold_t_nonword, drift_word_t, drift_nonword_t);
}

generated quantities {
    vector[N] log_lik;
    {
        for (n in 1:N){
            log_lik[n] = race_lpdf(block(RT, n, 1, 1, 2)| segment(ndt_t, n, 1), segment(threshold_t_word, n, 1),
                                   segment(threshold_t_nonword, n, 1), segment(drift_word_t, n, 1),
                                   segment(drift_nonword_t, n, 1));
        }
    }
}