


model {

// Priors
theta ~ std_normal()      // Standardized LV
for(i in 1:I) {
  for(c in 1:C) {
    // Note have to specify priors on the initial parameters; then they are
    // transformed in the transformed parameters block
    mu_init[i, c] ~ normal(mu_mean[i, c], Mu_cov[i, c, c])  
    lambda_init[i, c] ~ normal(lambda_mean[i, c], Lambda_cov[i, c, c])  
    }
  }

# Likelihood
  for(p in 1:P) {
    for(i in 1:I) {
      for(c in 1:C) {
        s = mu[i, c] + lambda[i, c] * theta[p] // Model for each category , item
        } 
        Y[i, p] ~ categorical_logit(s)  // "s": Simplex (probability vector)
      } 
    }
 }