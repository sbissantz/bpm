data{
  int<lower = 0> P;             // number of respondents 
  int<lower = 0> I;             // number of items
  int<lower = 0> C;             // number of categories
  // Important: The data are P x I, but we need I x P (row major order)
  array[I, P] int<lower = 1, upper = 5> Y;      // Item response array
  // Hyperparameters
  vector[I] lambda_mean;                  // prior means for lambda
  matrix[I, I] Lambda_cov;                // prior covmat for lambda
  array[I] vector[C-1] thr_mean;          // prior means for intercepts 
  array[I] matrix[C-1, C-1] Thr_cov;      // prior covmats for intercepts
}

parameters{
  vector[I] lambda;               // Loadings/discrimination/slopes  (1:item)
  array[I] ordered[C-1] thr;      //  Ord. vecs of intercepts (C-1:item)
  vector[P] theta;                // Latent trait (1:person)
}

model {
  // Priors
  lambda ~  multi_normal(lambda_mean, Lambda_cov);     // Prior p.d. loadings
  theta ~ std_normal();                                // Standardized LV
  for(i in 1:I) {
    thr[i] ~  multi_normal(thr_mean[i], Thr_cov[i]);   // Prior for intercepts 
    // Likelihood
    // Import: If we loop with '[i]' we access every person! (row major order)
    // So the statement is still vectorized because we do not loop over people
    Y[i] ~ ordered_logistic(lambda[i] * theta, thr[i]);
  }
}

generated quantities {
  array[I] vector[C-1] mu;      // Array of an ordered vector of intercepts
  for(i in 1:I) {
    mu[i] = -1*thr[i];              // From tresholds to intercepts
  }
}
