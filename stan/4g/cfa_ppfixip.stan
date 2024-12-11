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
  // Person parameters
  real theta_hypmean;               // prior hyper mean: theta
  real<lower=0> theta_hypsd;        // prior hyper sd: theta
  real<lower=0> theta_hyprate;      // prior hyper rate: theta
}
parameters { 
  // Item patameters
  vector[I] mu;               // item intercept (1:item)
  vector[I] lambda;           // factor loading/discrimination (1:item)
  vector<lower=0>[I] psi;     // factor loading/discrimination (1:item)
  // Person parameters
  vector[P] theta;             // empirical lv (1:person)
  real theta_mean;             // estimated prior mean: theta 
  real<lower=0> theta_sd;      // estimated prior sd: theta 
}
model {
  // Item parameters  
  mu ~ multi_normal(mu_mean, Mu_cov);             // item intercept prior p.d.
  lambda ~ multi_normal(lambda_mean, Lambda_cov); // factor loading prior p.d.
  psi ~ exponential(psi_rate);
  // Person parameters
  theta_mean ~ normal(theta_hypmean, theta_hypsd);
  theta_sd ~ exponential(theta_hyprate);
  theta ~ normal(theta_mean, theta_sd);
  // Likelihood
  for (i in 1:I) {
    // Vectorize over person (across rows)
    // Y[i] ~ normal(mu[i] + lambda[i] * theta, psi);
    Y[,i] ~ normal(mu[i] + lambda[i] * theta, psi[i]);
  }
}
