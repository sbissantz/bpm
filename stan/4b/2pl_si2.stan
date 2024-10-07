data {
  int<lower=0> P;                       // number of observations
  int<lower=0> I;                       // number of items
  array[I,P] int<lower=0, upper=1> Y;   // item response array 
  // Hyperparameters
  vector[I] mu_mean;                   // prior mean: item intercepts
  matrix[I,I] Mu_cov;                  // prior covmat: item intecepts
  vector[I] lambda_mean;               // prior mean: loadings/discrimination 
  matrix[I,I] Lambda_cov;              // prior covmat: loadings/discrimination
}
parameters {
  // Person parameters  
  vector[P] theta;                   // standardized lv (1/person)
  // Item parameters  
  vector[I] mu;                      // item intercept  (1/item)
  vector[I] lambda;                  // standardized lv (1/item)
}
model {
  // Priors
  mu ~ multi_normal(mu_mean, Mu_cov);             // item intercepts
  lambda ~ multi_normal(lambda_mean, Lambda_cov); // loadings/discrimination 
  theta ~ normal(0, 1);                           // standardized lv 
  // Likelihood (Asm: conditional independence)
  for (i in 1:I) { 
    // Import: If we loop with '[i]' we access every person! (row major order)
    // So the statement is vectorized over person 
    Y[i] ~ bernoulli_logit(mu[i] + lambda[i] * theta);
  }
}
generated quantities {
  vector[I]  a;                     // dicrimination parameter
  vector[I]  b;                     // difficulty parameter
  for (i in 1:I) {
    a[i] = lambda[i];               // item discrimination
    b[i] = -1*(mu[i]/lambda[i]);    // item difficulty
  }
}
