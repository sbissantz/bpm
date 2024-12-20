data {
  int<lower = 0> P;             // Number of respondents 
  int<lower = 0> I;             // Number of items
  int<lower = 0> C;             // Number of categories
  int<lower = 0> D;             // Number of dimensions 
  // Input data 
  // Note: We have to define them as array, because the entries are integers
  array[I, P] int<lower=1, upper=C>  Y; // Array of item responses
  // Q-matrix
  // Note: We have to define them as array, because the entries are integers
  array[I, D] int<lower=0, upper=1> Q; // Q-matrix
  // Hyperparameters
  // Important: "I-D" lambdas beacause we need to set for each dimension a
  // lambda to a value of 1
  vector[I-D] lambda_mean;                // Prior means for lambda
  matrix[I-D, I-D] Lambda_cov;            // Prior covmat for lambda
  array[I] vector[C-1] thr_mean;          // Prior means for intercepts 
  array[I] matrix[C-1, C-1] Thr_cov;      // Prior covmats for intercepts
  vector[D] theta_mean;                   // Prior means for theta 
  // New: Lognormal prior for the standard deviation of theta
  // Lognormal has two components: location parameter and scale parameter
  vector[D] theta_sd_loc;               // Hyperprior for theta_sd: location 
  vector[D] theta_sd_scl;               // Hyperprior for theta_sd: scale 
}

transformed data {
  // Important: If every item loads on every dimension, then L = I; but with
  // multidimensional items, L > I, items can load on multiple dimensions
  int<lower = 0> L = 0;             // Number of loadings (max. D:item)
  // Marker item 
  array[D] int<lower=0> i_marker = rep_array(0, D);

  // Count the number of loadings
  for(d in 1:D) {
    // Sum the non-zero row elements in each column of Q
    L = L + sum(Q[1:I, d]);
  }
  // Location of loadings in Q-matrix (row, column), plus marker switch
  // todo todo
  array[L, 4] int ll_Q;      
  int load_no = 1;               // Initial loading number, here: 1
  int lambda_no = 1;             // Initial lambda number, here: 1
  for (i in 1:I){
    for (d in 1:D){
      // the value in the Q-matrix is 1
      if (Q[i, d] == 1){
        // "Loading location" in the Q-Matrix: 
        // Save the row and column index of the loading
        ll_Q[load_no, 1] = i;
        ll_Q[load_no, 2] = d;
        if (i_marker[d] == 0){
          ll_Q[load_no, 3] = 1; // "1" if marker, "0" otherwise
          ll_Q[load_no, 4] = 0; // "0" if not one of estimated lambdas
          i_marker[d] = i; 
        } else {
          ll_Q[load_no, 3] = 0;
          ll_Q[load_no, 4] = lambda_no;
          lambda_no = lambda_no + 1;
        }
        // Increase the loading number
        load_no = load_no + 1;
      }
    }
  }
}

parameters {
  array[I] ordered[C-1] thr;           // Ord. vecs of intercepts (C-1:item)
  vector[L-D] lambda_init;             // Loadings vector (L-D:item)
  array[P] vector[D] theta;            // Latent trait (D:person)
  // Note: Use cholesky factor for numerical stability 
  cholesky_factor_corr[D] L_cor_theta; // Cholesky factor of Cor_theta 
  vector<lower=0>[D] theta_sd;         // Standard deviation of thetas
}

transformed parameters {
  matrix[I, D] Lambda = rep_matrix(0.0, I, D); // Initialize the Q matrix
  matrix[P, D] Theta;           // Matrix of standardized LVs (P x D)
  // Build the Lambda matrix to multiply with the Theta matrix
  for (l in 1:L){
    if (ll_Q[l,3] == 1){
      Lambda[ll_Q[l,1], ll_Q[l,2]] = 1.0;
    } else { 
      Lambda[ll_Q[l,1], ll_Q[l,2]] = lambda_init[ll_Q[l,4]]; 
    }
  }
  for (d in 1:D){
    // Type conversion: Theta needs to be a matrix to multiply it with Lambda, 
    // but we initialized it as an array of vectors
    Theta[,d] = to_vector(theta[,d]);
  }
}

model { 
  matrix[D, D] L_cov_theta;
  L_cov_theta = diag_pre_multiply(theta_sd, L_cor_theta);
  // Priors
  lambda_init ~ multi_normal(lambda_mean, Lambda_cov); 
  L_cor_theta ~ lkj_corr_cholesky(1.0);    
  theta_sd ~ lognormal(theta_sd_loc, theta_sd_scl);
  theta ~ multi_normal_cholesky(theta_mean, L_cor_theta);    
  // Likelihood 
  for (i in 1:I) { 
    thr[i] ~  multi_normal(thr_mean[i], Thr_cov[i]);   // Prior for intercepts 
    // Import: If we loop with '[i]' we access every person! (row major order)
    // So the statement is still vectorized because we do not loop over people 
    // Note: Lambda * Theta' = Theta'' * Lambda' = Theta * Lambda'
    Y[i] ~ ordered_logistic(Theta * Lambda[i, 1:D]', thr[i]);
    // Import: Theta is now a P x D matrix 
  }
}

generated quantities {
  array[I] vector[C-1] mu;      // Array of an ordered vector of intercepts
  // From ordered thresholds to inversly ordered intercepts
  for(i in 1:I) {
    mu[i] = -1*thr[i];              
  }
  // Important: We can usually ignore the covariance. It is to hard to interpret
  // and we are usually interessted in the correlations 
  cholesky_factor_cov[D] Theta_cov_pre;
  cov_matrix[D] Theta_cov;
  corr_matrix[D] Theta_cor;  // Between factor correlation 
  // Calculate the between factor correlation from the Cholesky factor
  // Note: Theta_cor = L_cor_theta * L_cor_theta' 
  Theta_cor = multiply_lower_tri_self_transpose(L_cor_theta);
  Theta_cov_pre = diag_pre_multiply(theta_sd, L_cor_theta); 
  Theta_cov = multiply_lower_tri_self_transpose(Theta_cov_pre); 
}
