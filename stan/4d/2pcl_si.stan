
data { 
  int<lower = 0> P;             // Number of respondents 
  int<lower = 0> I;             // Number of items
  int<lower = 0> C;             // Number of categories 
  // Important: The data are P x I, but we need I x P (row major order)
  array[I, P] int<lower=1, upper= C> Y; // Item response array
  // Hyperparameters
  array[I] vector[C-1] mu_mean;          // Prior means for intercepts
  array[I] matrix[C-1, C-1] Mu_cov;      // Prior covmats for intercepts
  array[I] vector[C-1] lambda_mean;      // Prior means for loadings 
  array[I] matrix[C-1, C-1] Lambda_cov;  // Prior covmats for loadings
}

parameters {
  array[I] vector[C-1]  mu_init;      // Inital intercepts (C-1:item)
  array[I] vector[C-1]  lambda_init;  // Initial loadings (C-1:item)
  vector[P] theta;                    // Latent variable (1:person)
}

transformed parameters {
  array[I] vector[C]  mu;      // Actual intercepts (C-1:item)
  array[I] vector[C]  lambda;  // Actual loadings (C-1:item)
  for (i in 1:I) {
    mu[i, 1] = 0.0;     // Set first intercept to zero
    mu[i, 2:C] = mu_init[i, 1:(C-1)];
    lambda[i, 1] = 0.0; // Set first loading to zero
    lambda[i, 2:C] = lambda_init[i, 1:(C-1)];
  }
}

model {
vector[C] s; // s: Simplex (probability vector)
// Priors
theta ~ std_normal();      // Standardized LV
for(i in 1:I) {
  for(c in 1:(C-1)) {
    // Note have to specify priors on the initial parameters; then they are
    // transformed in the transformed parameters block
    mu_init[i, c] ~ normal(mu_mean[i, c], Mu_cov[i, c, c]);  
    lambda_init[i, c] ~ normal(lambda_mean[i, c], Lambda_cov[i, c, c]);
    }
  }
// Likelihood
  for(p in 1:P) {
    for(i in 1:I) {
      for(c in 1:C) {
        s[c] = mu[i, c] + lambda[i, c] * theta[p]; // Model 1:category, item
        } 
        Y[i, p] ~ categorical_logit(s);  
      } 
    }
 }
