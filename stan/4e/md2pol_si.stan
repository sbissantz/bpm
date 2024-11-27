data {
  int<lower = 0> P;             // Number of respondents 
  int<lower = 0> I;             // Number of items
  int<lower = 0> C;             // Number of categories
  int<lower = 0> D;             // Number of dimensions 
  // Important: If every item loads on every dimension, then L = I; but with
  // multidimensional items, L > I, items can load on multiple dimensions
  int<lower = 0> L;             // Number of loadings (for m-D items)
}

parameters {
  matrix[P, D] Theta;
  vector[L] lambda;             // Number of loadings
  array[I] ordered[C-1] thr; 
}

model {

for (i in 1:I) { 
  thr[i] ~  multi_normal(thr_mean[i], Thr_cov[i]);   // Prior for intercepts 
  // Likelihood
  // Import: If we loop with '[i]' we access every person! (row major order)
  // So the statement is still vectorized because we do not loop over people 
  Y[i] ~ ordered_logistic(lambda[i] * Theta, thr[i]);
  // Import: Theta is now a matrix with dimension P x D
  }
}

generated quantities {
  array[I] vector[C-1] mu;      // Array of an ordered vector of intercepts
  for(i in 1:I) {
    mu[i] = -1*thr[i];              // From tresholds to intercepts
  }
}
