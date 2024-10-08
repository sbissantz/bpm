data {
  int<lower=0> P;    // number of persons
  int<lower=0> I;     // number of items
  // Important: The data are P x I, but we need I x P (row major order)
  array[I, P] int<lower=0, upper=1> Y;      // item responses
  // Hyperparamers
  vector[I] a_mean;
  matrix[I, I] A_cov;
  vector[I] b_mean;
  matrix[I, I] B_cov;
}
parameters {
  // Item  parameters
  vector[I] a;      // item discrimination (1item)
  vector[I] b;      // item difficulty (1:item)
  // Person parameters
  vector[P] theta;      // latent variable (1:person) 
}
model {
  // Item parameters
  a ~ multi_normal(a_mean, A_cov);     // item discrimination prior 
  b ~ multi_normal(b_mean, B_cov);     // item difficulty prior 
  // Person parameters
  theta ~ std_normal();      // standardized lv priors 
  // Likelihood
  for (i in 1:I) {
  // Import: If we loop with '[i]' we access every person! (row major order)
  // So the statement is vectorized over person 
    Y[i] ~ bernoulli(Phi_approx(a[i] * (theta - b[i])));
    // Y[i] ~ bernoulli(Phi(a[i] * (theta - b[i])));
  }
}
