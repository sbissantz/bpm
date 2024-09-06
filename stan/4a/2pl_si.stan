data {
  int<lower=0> P;   // number of observations
  int<lower=0> I;   // number of items
  
  // Important: The data are P x I, but we need I x P (row major order)
  array[I, P] int<lower=0, upper=1>  Y; // item responses in an array

  vector[I] mu_mean;        // prior mean vector for intercept parameters
  matrix[I, I] Mu_cov;      // prior covmat for intercept parameters
  
  vector[I] lambda_mean;    // prior mean vector for discrimination parameters
  matrix[I, I] Lambda_cov;  // prior covmat for discrimination parameters
}

parameters {
  vector[P] theta;    // latent variables (one for each person)
  vector[I] mu;       //  item intercepts (one for each item)
  vector[I] lambda;   // factor loadings (one for each item)
}

model { 
  // Prior for item discrimination/factor loadings
  lambda ~ multi_normal(lambda_mean, Lambda_cov); 

  mu ~ multi_normal(mu_mean, Mu_cov); // Prior for item intercepts
  theta ~ normal(0, 1);               // Prior for LV (with mean/sd specified)
  for (i in 1:I){
    
    // Import: If we loop with '[i]' we access every person! (row major order)
    // So the statement is still vectorized because we do not loop over people
    Y[i] ~ bernoulli_logit(mu[i] + lambda[i]*theta);
  }
}
