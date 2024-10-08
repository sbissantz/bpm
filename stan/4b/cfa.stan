data {
  int<lower = 0> P;         // number of persons
  int<lower = 0> I;         // number of items
  matrix[P, I] Y;           // item response matrix
  // array[I, P] real Y;    // item response array 
  // Hyperparameters
  vector[I] mu_mean;        // prior mean: item intercept 
  matrix[I,I] Mu_cov;       // prior covmat: item intercept 
  vector[I] lambda_mean;    // prior mean: factor loading
  matrix[I,I] Lambda_cov;   // prior covmat: factor loading
  vector[I] psi_rate;       // prior rate: unique std
}
parameters { 
  // Item patameters
  vector[I] mu;               // item intercept (1:item)
  vector[I] lambda;           // factor loading/discrimination (1:item)
  vector<lower=0>[I] psi;     // factor loading/discrimination (1:item)
  // Person parameters
  vector[P] theta;             // standardized lv (1:person)
}
model {
  // Item parameters  
  mu ~ multi_normal(mu_mean, Mu_cov);              // item intercept prior p.d.
  lambda  ~ multi_normal(lambda_mean, Lambda_cov); // factor loading prior p.d.
  psi ~ exponential(psi_rate);
  // Person parameters
  theta ~ std_normal();
  // Likelihood
  for (i in 1:I) {
    // Vectorize over person (across rows)
    // Y[i] ~ normal(mu[i] + lambda[i] * theta, psi);
    Y[,i] ~ normal(mu[i] + lambda[i] * theta, psi[i]);
  }
}
