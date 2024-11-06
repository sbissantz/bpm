data{
  int<lower = 0> P;             // Number of respondents 
  int<lower = 0> I;             // Number of items
  int<lower = 0> C;             // Number of categories
  array[I, P] int<lower = 1, upper = 5> Y;           // Item response array
  // Hyperparameters
  vector[I] lambda_mean;                  // Mean for lambda
  matrix[I, I] Lambda_cov;                // Covmat for lambda
  array[I] vector[C-1] thr_mean;          // Mean for intercepts 
  array[I] matrix[C-1, C-1] Thr_cov;      // Covmat for intercepts
}
parameters{
  vector[I] lambda;               // Loadings/discrimination/slopes  (1:item)
  array[I] ordered[C-1] thr;      // Array of ord. vecs of intercepts (1:item)
  vector[P] theta;                // Latent trait (1:person)
}
model {
  // Priors
  lambda ~  multi_normal(lambda_mean, Lambda_cov);    // Prior p.d. loadings
  theta ~ std_normal();                                // Standardized LV
  for(i in 1:I) {
    thr[i] ~  multi_normal(thr_mean[i], Thr_cov[i]);
    // Likelihood
    // Vectorized over persons
    Y[i] ~ ordered_logistic(lambda[i] * theta, thr[i]);
  }
}
generated quantities {
  array[I] vector[C-1] mu;      // Array of an ordered vector of intercepts
  for(i in 1:I) {
    mu[i] = -1*thr[i];              // From tresholds to intercepts
  }
}
