data {
  int<lower=0>  P;  // number of respondents 
  int<lower=0>  I;  // number of items
  array[I, P] int<lower=0, upper=1> Y;
  // Hyperparameters
  vector[I] a_mean; // item discirimination: mean vec
  matrix[I,I] A_cov; // item discirimination: covmat
  vector[I] b_mean; // item difficulty: mean vec
  matrix[I,I] B_cov; // item difficulty: covmat
}
parameters {
  // Item parameters
  vector[I] a; // item discimination
  vector[I] b; // item difficulty
  // Person parameters
  vector[P] theta; // person ability/latent trait
}
model{
  // Priors
  a ~ multi_normal(a_mean, A_cov); // item discirimination
  b ~ multi_normal(b_mean, B_cov); // item difficulty 
  theta ~ std_normal(); // standardized lv
  // Likelihood
  // Import: If we loop with '[i]' we access every person! (row major order)
  // So the statement is vectorized over person 
  for (i in 1:I) {
    Y[i] ~ bernoulli_logit(a[i] * (theta - b[i])); 
  }
}
generated quantities {
 vector[I] mu;
 vector[I] lambda;
  // Slope / intercept 
  for (i in 1:I) {
    mu[i] = -1 * a[i] * b[i]; // item intercept
  }
  lambda = a; // factor loading/item discirimination 
}
