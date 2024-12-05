data {
  int<lower = 0> P;             // Number of respondents 
  int<lower = 0> I;             // Number of items
  int<lower = 0> C;             // Number of categories
  int<lower = 0> D;             // Number of dimensions 
  // Missing data
  array[I] int<lower = 0> n_obs;      // Number of observed values (1:item)
  array[I, P] int Obs_idx;            // Index of observed values (I x P)

  // Input data 
  // Note: We have to define them as array, because the entries are integers
  array[I, P] int<lower=-1, upper=C>  Y; // Array of item responses
  // Q-matrix
  // Note: We have to define them as array, because the entries are integers
  array[I, D] int<lower=0, upper=1> Q; // Q-matrix
  // Hyperparameters
  vector[I] lambda_mean;                  // Prior means for lambda
  matrix[I, I] Lambda_cov;                // Prior covmat for lambda
  array[I] vector[C-1] thr_mean;          // Prior means for intercepts 
  array[I] matrix[C-1, C-1] Thr_cov;      // Prior covmats for intercepts
  vector[D] theta_mean;                   // Prior means for theta 
}

transformed data {
  // Important: If every item loads on every dimension, then L = I; but with
  // multidimensional items, L > I, items can load on multiple dimensions
  int<lower = 0> L = 0;             // Number of loadings (max. D:item)
  // Count the number of loadings
  for(d in 1:D) {
    // Sum the non-zero row elements in each column of Q
    L = L + sum(Q[1:I, d]);
  }
  array[L, 2] int ll_Q;      // Location of loadings in Q-matrix (row, column)
  int lno = 1;               // Initial loading number, here: 1
  for (i in 1:I){
    for (d in 1:D){
      // the value in the Q-matrix is 1
      if (Q[i, d] == 1){
        // Save the row and column index of the loading
        ll_Q[lno, 1] = i;
        ll_Q[lno, 2] = d;
        // Increase the loading number
        lno = lno + 1;
      }
    }
  }
}

parameters {
  array[I] ordered[C-1] thr;           // Ord. vecs of intercepts (C-1:item)
  vector[L] lambda;                    // Loadings vector (max D:item)
  array[P] vector[D] theta;            // Latent trait (D:person)
  // Note: Use cholesky factor for numerical stability 
  cholesky_factor_corr[D] L_cor_theta; // Cholesky factor of Cor_theta 
  
}

transformed parameters {
  matrix[I, D] Lambda = rep_matrix(0.0, I, D); // Initialize the Q matrix
  matrix[P, D] Theta;           // Matrix of standardized LVs (P x D)
  // Build the Lambda matrix to multiply with the Theta matrix
  for (l in 1:L){
    Lambda[ll_Q[l,1], ll_Q[l,2]] = lambda[l];
  }
  for (d in 1:D){
    // Type conversion: Theta needs to be a matrix to multiply it with Lambda, 
    // but we initialized it as an array of vectors
    Theta[,d] = to_vector(theta[,d]);
  }
}

model { 
  // Priors
  lambda ~ multi_normal(lambda_mean, Lambda_cov); 
  L_cor_theta ~ lkj_corr_cholesky(1.0);    
  theta ~ multi_normal_cholesky(theta_mean, L_cor_theta);    
  // Likelihood 
  for (i in 1:I) { 
    thr[i] ~  multi_normal(thr_mean[i], Thr_cov[i]);   // Prior for intercepts 
    // Import: If we loop with '[i]' we access every person! (row major order)
    // So the statement is still vectorized because we do not loop over people 
    // Note: Lambda * Theta' = Theta'' * Lambda' = Theta * Lambda'
    // ----------------------------------------------
    // Note: With missing values, we simply loop over the observed values,
    // ASM: MAR assumption (or listwise deletion)
    // That the lhs and the rhs have the same n° elements, we index also the lhs
    Y[i, Obs_idx[i, 1:n_obs[i]]] ~ ordered_logistic(Theta[Obs_idx[i, 1:n_obs[i]]] * Lambda[i, 1:D]', thr[i]);
    // Import: Theta is now a P x D matrix 
  }
}

generated quantities {
  array[I] vector[C-1] mu;      // Array of an ordered vector of intercepts
  // From ordered thresholds to inversly ordered intercepts
  for(i in 1:I) {
    mu[i] = -1*thr[i];              
  }
  corr_matrix[D] Theta_cor;  // Between factor correlation 
  // Calculate the between factor correlation from the Cholesky factor
  // Note: Theta_cor = L_cor_theta * L_cor_theta' 
  Theta_cor = multiply_lower_tri_self_transpose(L_cor_theta);
}
