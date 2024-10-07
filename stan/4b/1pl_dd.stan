data {
  int<lower=0>  P;  // number of respondents 
  int<lower=0>  I;  // number of items
  array[I, P] int<lower=0, upper=1> Y;
  // Hyperparameters
  vector[I] b_mean; // item difficulty: mean vec
  matrix[I,I] B_cov; // item difficulty: covmat
}
parameters {
  // Item parameters
  vector[I] b; // item difficulty
  // Person parameters
  vector[P] theta; // person ability/latent trait
}
model{
  // Priors
  b ~ multi_normal(b_mean, B_cov); // item difficulty 
  theta ~ std_normal(); // standardized lv
  // Likelihood
  // Import: If we loop with '[i]' we access every person! (row major order)
  // So the statement is vectorized over person 
  for (i in 1:I) {
    // Y[i] ~ bernoulli(inv_logit(theta - b[item]));
    Y[i] ~ bernoulli_logit(theta - b[i]); 
  }
}
