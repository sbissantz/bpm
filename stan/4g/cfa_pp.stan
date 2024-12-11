// Adaptively regularized CFA model
// Empirirical priors on item intercepts and factor loadings 
data {
  int<lower = 0> P;         // number of persons
  int<lower = 0> I;         // number of items
  matrix[P, I] Y;           // item response matrix
  // array[I, P] real Y;    // item response array 
  // Item parameters 
  real mu_hypmean;                  // prior hyper mean: item intercept
  real<lower=0> mu_hypsd;           // prior hyper sd: item intercept
  real<lower=0> mu_hyprate;         // prior hyper rate: item intercept
  real lambda_hypmean;              // prior hyper mean: item intercept
  real<lower=0> lambda_hypsd;       // prior hyper mean: item intercept
  real<lower=0> lambda_hyprate;     // prior hyper mean: item intercept
  real<lower=0> psi_hyprate;        // prior hyper rate: unique std 
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
  // Item hyper patameters
  real mu_mean;               // estimated prior mean: item intercept 
  real<lower=0> mu_sd;        // estimated prior sd: item intercept 
  real lambda_mean;           // estimated prior mean: factor loading 
  real<lower=0> lambda_sd;    // estimated prior sd: factor loading 
  real<lower=0> psi_rate;     // estimated prior rate: unique std
  // Person parameters
  vector[P] theta;             // empirical lv (1:person)
  real theta_mean;             // estimated prior mean: theta 
  real<lower=0> theta_sd;      // estimated prior sd: theta 
}
model {
  // Item parameters  
  mu ~ normal(mu_mean, mu_sd);              // item intercept prior p.d.
  mu_mean ~ normal(mu_hypmean, mu_hypsd);   // hyper prior p.d.  
  mu_hypsd ~ exponential(mu_hyprate);       // hyper prior SD 
  lambda ~ normal(lambda_mean, lambda_sd); // factor loading prior p.d.
  lambda_mean ~ normal(lambda_hypmean, lambda_hypsd); // hyper prior p.d.
  lambda_hypsd ~ exponential(lambda_hyprate);       // hyper prior SD 
  psi ~ exponential(psi_rate);        // unique std prior p.d.
  psi_rate  ~ exponential(psi_hyprate);
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
