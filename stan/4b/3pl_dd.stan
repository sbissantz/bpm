data {
  int<lower = 0> P;
  int<lower = 0> I;
  array[I, P] int<lower = 0, upper = 1> Y;
  // Hyperparameter
  vector[I] a_mean;       // item discrimination: mean vec
  matrix[I,I] A_cov;      // item discrimination: covmat
  vector[I] b_mean;       // item difficulty: mean vec
  matrix[I,I] B_cov;      // item difficulty: covmat
}
parameters {
  // Item parameters
  vector[I] a;      // item discrimination (1:item)
  vector[I] b;      // item difficulty (1:item)
  vector<lower = 0, upper = 1>[I] c;      // pseudo guessing parameter (1:item)
  // Person parameters
  vector[P] theta;      // person parameter (1:person)
}
model{
  // Priors
  a ~ multi_normal(a_mean, A_cov);     // item discrimination
  b ~ multi_normal(b_mean, B_cov);     // item difficulty
  // Note: Since c is a vector we get I independent beta p.d.s. We could also
  // use a multivariate beta, i.e. a dirichlet p.d. with alpha equal to 1
  c ~ beta(1, 1);
  theta ~ std_normal();      // standardized lv 
  // Likelihood
  // Import: If we loop with '[i]' we access every person! (row major order)
  // So the statement is vectorized over person 
  for (i in 1:I) {
    Y[i] ~ bernoulli(c[i] + (1-c[i]) * inv_logit(a[i]*(theta - b[i])));
  }
}
