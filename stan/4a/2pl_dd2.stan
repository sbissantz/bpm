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
  vector[P]  theta;      // Latent trait
}
model {
  // Prior p.d.s
  a ~ multi_normal(a_mean, A_cov);    // item discrimination 
  b ~ multi_normal(b_mean, B_cov);    // item difficulty
  // Person parameters
  theta ~ std_normal();               // standardized LV
  // Likelihood (ASM: conditional indep. Person/Items)
  for (i in 1:I) {
    // Import: If we loop with '[i]' we access every person! (row major order)
    Y[i] ~ bernoulli_logit(a[i] * (theta - b[i]));
  }
}
generated quantities {
  // Item parameters
  vector[I] mu;         // item intercept
  vector[I] lambda;     // factor loading
  // Auxiliary statistics 
  vector[N_theta] TCC;              // test characteristic curve
  matrix[N_theta, I] item_info;     // item information function
  vector[N_theta] test_info;        // test information function
  // Factor loading 
  lambda = a;
  // Auxiliary statistics (setup) 
  TCC = rep_vector(0, N_theta);
  // Test info must start at -1 to include prior p.d. for theta
  test_info = rep_vector(-1, N_theta);
  // Item info must start at -1 to include prior p.d. for theta
  item_info = rep_matrix(-1, N_theta, I);
  for (i in 1:I) {
    mu[i] = -1*a[i]*b[i];     // Item intercept 
    for (v in 1:N_theta) {
      // Test characteristic curve
      // TCC[v] = TCC[v] + inv_logit(a[i]*(theta_fix[v]-b[i]));
      TCC[v] += inv_logit(a[i]*(theta_fix[v]-b[i]));     // compund
      // Item information functions
      // item_info[v,i] = item_info[v,i] + a[i]^2*inv_logit(a[i]*(theta_fix[v]-b[i]))*(1-inv_logit(a[i]*(theta_fix[v]-b[i])));
      item_info[v,i] += a[i]^2 * inv_logit(a[i]*(theta_fix[v]-b[i]))*(1-inv_logit(a[i]*(theta_fix[v]-b[i])));     // compound
      // Test information functions
      // test_info[v] = test_info[v] + a[i]^2*inv_logit(a[i]*(theta_fix[v]-b[i]))*(1-inv_logit(a[i]*(theta_fix[v]-b[i])));
      test_info[v] += a[i]^2*inv_logit(a[i]*(theta_fix[v]-b[i]))*(1-inv_logit(a[i]*(theta_fix[v]-b[i])));
    }
  }
}
