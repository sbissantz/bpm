data {
  int<lower = 0> P;             // number of respondents 
  int<lower = 0> I;             // number of items
  int<lower = 0> C;             // number of categories
  // Important: The data are P x I, but we need I x P (row major order)
  array[I, P] int<lower = 1, upper = 5> Y;      // Item response array
  // Hyperparameters
  vector[I-1] lambda_mean;                // prior means for lambda
  matrix[I-1, I-1] Lambda_cov;            // prior covmat for lambda
  array[I] vector[C-1] thr_mean;          // prior means for intercepts 
  array[I] matrix[C-1, C-1] Thr_cov;      // prior covmats for intercepts
  real<lower=0> theta_hypermean;          // Hyperprior for theta SD
  real<lower=0> theta_hypersd;            // Hyperprior for theta SD
}

parameters {
  vector[I-1] lambda_init;        // Loadings for estimated(!) items 
  // Note: I-1; the minus one comes from the marker item, which is fixed
  array[I] ordered[C-1] thr;      // Ord. vecs of intercepts (C-1:item)
  vector[P] theta;                // Latent trait (1:person)
  real<lower=0> theta_sd;         // Estimated SD of theta 
}

transformed parameters {
  vector[I] lambda;               // Loadings/discrimination/slopes (1:item)
  lambda[1] = 1.0;                // First loading fixed to 1 (often default) 
  lambda[2:I] = lambda_init[1:(I-1)]; // Rest is set to estimated items 
}

model {
  // Priors
  // Prior p.d. for estimated(!) loadings – I - 1 (marker item)
  lambda_init ~  multi_normal(lambda_mean, Lambda_cov); 
  // Latent variable (non-standardized) - estimate theta_sd
  theta ~ normal(0, theta_sd);                         
  // Hyperprior for estimated factor SD                        
  theta_sd ~ lognormal(theta_hypermean, theta_hypersd); 
  for(i in 1:I) {
    thr[i] ~  multi_normal(thr_mean[i], Thr_cov[i]);   // Prior for intercepts 
    // Likelihood
    // Import: If we loop with '[i]' we access every person! (row major order)
    // So the statement is still vectorized because we do not loop over people
    Y[i] ~ ordered_logistic(lambda[i] * theta, thr[i]);
  }
}

generated quantities {
  array[I] vector[C-1] mu;      // Array of a vector of intercepts
  for(i in 1:I) {
    mu[i] = -1*thr[i];              // From tresholds to intercepts
  }
}
