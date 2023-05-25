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
    vector<lower=0>[N] threshold_word;
    vector<lower=0>[N] threshold_nonword;
    vector<lower=0>[N] k_1;
    vector<lower=0>[N] k_2;
    vector<lower=0>[N] alpha;
    vector<lower=0>[N] b;
    vector<lower=0>[N] g;
                                            
    vector[2] m_priors;
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
    real<lower=0> m;
}

transformed parameters {
    vector<lower=0>[N] drift_word_t;                     // trial-by-trial drift rate for predictions
    vector<lower=0>[N] drift_nonword_t;                  // trial-by-trial drift rate for predictions
    vector<lower=0>[N] ndt_t;                  // trial-by-trial drift rate for predictions

    for (n in 1:N) {
        drift_word_t[n] = (k_1[n] + b[n] * frequency[n]) / (1 + exp(-alpha[n] * (p[n][1]-0.5)));
        drift_nonword_t[n] = k_2[n] / (1 + exp(-alpha[n] * (p[n][2]-0.5)));

        ndt_t[n] = (m +  g[n] * exp(-frequency[n])) * (minRT[n] - RTbound) + RTbound;
    }
}

model {
    m ~ normal(m_priors[1], m_priors[2]);
    
    RT ~ race(ndt_t, threshold_word, threshold_nonword, drift_word_t, drift_nonword_t);
}

generated quantities {
    vector[N] log_lik;
    {
        for (n in 1:N){
            log_lik[n] = race_lpdf(block(RT, n, 1, 1, 2)| segment(ndt_t, n, 1), segment(threshold_word, n, 1),
                                   segment(threshold_nonword, n, 1), segment(drift_word_t, n, 1),
                                   segment(drift_nonword_t, n, 1));
        }
    }
}