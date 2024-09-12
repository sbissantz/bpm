data {
  int<lower=0> P;                      // number of persons 
  int<lower=0> I;                      // number of items
  array[I, P] int<lower=0, upper=1> Y; // item response array
  // Prior hyperparameters
  vector[I] a_mean;                // item discrimination: mean
  matrix[I, I] A_cov;              // item discrimination: covmat
  vector[I] b_mean;                 // item difficulty: mean
  matrix[I, I] B_cov;               // item difficulty: covmat
  // Auxiliary statistics
  int<lower=0> N_theta;                 // number of fixed thetas 
  vector[N_theta] theta_fix;            // fixed theta values   
}
parameters {
  // Item parameters 
  vector[I]  a;     // item discrimination/loading
  vector[I]  b;      // item difficulty
  // Person parameters 
  vector[I]  theta;      // Latent trait
}
model {
  // Prior p.d.s
  a ~ multi_normal(a_mean, A_cov); // item discrimination 
  b ~ multi_normal(b_mean, B_cov);    // item difficulty
  // Person parameters
  theta ~ standard_normal();                   // standardized LV
  // Likelihood (ASM: conditional indep. Person/Items)
  for (i in 1:I) {
    // Import: If we loop with '[i]' we access every person! (row major order)
    Y[i] ~ bernoulli_logit(a[i] * (theta - b[i]))
  }
}
generated quantities {
  // Item parameters
  vector[I] mu;         // item intercept
  vector[I] lambda;     // factor loading
  // Auxiliary statistics 
  vector[N_theta] ICC;              // item characteristic curve
  vector[N_theta] TCC;              // test characteristic curve
  matrix[N_theta, I] item_info;     // item information function
  vector[N_theta] test_info;        // test information function

  // Item intercept 
  for (i in 1:I) {
    mu[i] = -1*a[i]*b[i];
  }
  // Factor loading 
  lambda = a;
  // TODO: Test characteristic curve
}
