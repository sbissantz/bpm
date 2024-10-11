data {
  int<lower = 0> P;         // number of persons
  int<lower = 0> I;         // number of items
  matrix[P, I] Y;           // item response matrix
  // array[I, P] real Y;    // item response array 
  // Fixed Item patameters
  vector[I] mu_eap;               // fixed item intercept
  vector[I] lambda_eap;           // fixed factor loading/discrimination
  vector<lower=0>[I] psi_eap;     // fixedfactor loading/discrimination
}
parameters { 
  // Person parameters
  vector[P] theta;             // latent trait 
}
model {
  // Person parameters
  theta ~ std_normal();                           // standardized lv
  // Likelihood
  for (i in 1:I) {
    // Vectorize over person (across rows)
    // Y[i] ~ normal(mu[i] + lambda[i] * theta, psi);
    Y[,i] ~ normal(mu_eap[i] + lambda_eap[i] * theta, psi_eap[i]);
  }
}

