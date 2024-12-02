
#########
# Setup #
#########

library(cmdstanr)
  # set number of cores to 4 for this analysis
  options(mc.cores = 4)
library(bayesplot)
library(ggplot2)
library(posterior)

# Model the belief in conspiracy theories,
conspiracy_data <- read.csv("./data/conspiracies.csv")

# Only using the first 10 items
# Positive values mean resemble agreement
citems <- conspiracy_data[, 1 : 10]

# Number of items 
I <- ncol(citems)

# Number of respondents
P <- nrow(citems)

# Number of categories
C <- max(citems)

# Number of dimensions
D <- 2
# Test if the data are conformable with two factors: 
# (1) A government factor & (2) a non-goverment factor 

############
# Q-Matrix #
############

# Initialize a I X D matrix with zeros
Q <- matrix(0, nrow = I, ncol = D)
colnames(Q) <- c("Gov", "NonGov")
rownames(Q) <- paste0("i", seq_len(I))
# Determine which items measure each factor
# Factor 1 is measured by (loads on) items...
itms_d1 <- c(2, 5, 7, 8, 9)
Q[itms_d1, 1] <- 1
# Factor 2 is measured by (loads on) items...
itms_d2 <- c(1, 3, 4, 6, 10)
Q[itms_d2, 2] <- 1

# Import: We have no multidimensional items, every factor is measured by a
# unique set of items.
Q

# Ordered Logit (Multinomial/categorical distribution) Model Syntax =======================


# Compile model
mdl_md2polsi <- cmdstan_model("./stan/4e/md2pol_si.stan", pedantic = TRUE)

# Stan model for the multi-dimensional ordered logistic model

modelOrderedLogit_syntax = "
data {
  
  // data specifications  =============================================================
  int<lower=0> nObs;                            // number of observations
  int<lower=0> nItems;                          // number of items
  int<lower=0> maxCategory;       // number of categories for each item
  
  // input data  =============================================================
  array[nItems, nObs] int<lower=1, upper=5>  Y; // item responses in an array

  // loading specifications  =============================================================
  int<lower=1> nFactors;                                       // number of loadings in the model
  array[nItems, nFactors] int<lower=0, upper=1> Qmatrix;
  
  // prior specifications =============================================================
  array[nItems] vector[maxCategory-1] meanThr;                // prior mean vector for intercept parameters
  array[nItems] matrix[maxCategory-1, maxCategory-1] covThr;  // prior covariance matrix for intercept parameters
  
  vector[nItems] meanLambda;         // prior mean vector for discrimination parameters
  matrix[nItems, nItems] covLambda;  // prior covariance matrix for discrimination parameters
  
  vector[nFactors] meanTheta;
}

transformed data{
  int<lower=0> nLoadings = 0;                                      // number of loadings in model
  
  for (factor in 1:nFactors){
    nLoadings = nLoadings + sum(Qmatrix[1:nItems, factor]);
  }

  array[nLoadings, 2] int loadingLocation;                     // the row/column positions of each loading
  int loadingNum=1;
  
  for (item in 1:nItems){
    for (factor in 1:nFactors){
      if (Qmatrix[item, factor] == 1){
        loadingLocation[loadingNum, 1] = item;
        loadingLocation[loadingNum, 2] = factor;
        loadingNum = loadingNum + 1;
      }
    }
  }


}

parameters {
  array[nObs] vector[nFactors] theta;                // the latent variables (one for each person)
  array[nItems] ordered[maxCategory-1] thr; // the item thresholds (one for each item category minus one)
  vector[nLoadings] lambda;             // the factor loadings/item discriminations (one for each item)
  cholesky_factor_corr[nFactors] thetaCorrL;
}

transformed parameters{
  matrix[nItems, nFactors] lambdaMatrix = rep_matrix(0.0, nItems, nFactors);
  matrix[nObs, nFactors] thetaMatrix;
  
  // build matrix for lambdas to multiply theta matrix
  for (loading in 1:nLoadings){
    lambdaMatrix[loadingLocation[loading,1], loadingLocation[loading,2]] = lambda[loading];
  }
  
  for (factor in 1:nFactors){
    thetaMatrix[,factor] = to_vector(theta[,factor]);
  }
  
}

model {
  
  lambda ~ multi_normal(meanLambda, covLambda); 
  thetaCorrL ~ lkj_corr_cholesky(1.0);
  theta ~ multi_normal_cholesky(meanTheta, thetaCorrL);    
  
  
  for (item in 1:nItems){
    thr[item] ~ multi_normal(meanThr[item], covThr[item]);            
    Y[item] ~ ordered_logistic(thetaMatrix*lambdaMatrix[item,1:nFactors]', thr[item]);
  }
  
  
}

generated quantities{ 
  array[nItems] vector[maxCategory-1] mu;
  corr_matrix[nFactors] thetaCorr;
   
  for (item in 1:nItems){
    mu[item] = -1*thr[item];
  }
  
  
  thetaCorr = multiply_lower_tri_self_transpose(thetaCorrL);
  
}

"

# Item threshold hyperparameter
Thr_mean <- replicate(C - 1, rep(0, I)) # 10 x 4
THR_cov <- array(0, dim = c(10, 4, 4)) # 10 x 4 x 4
for(d in seq_len(I)) {
  THR_cov[d , ,] <- diag(10, C - 1)
}

# Item discrimination/factor loading hyperparameters
lambda_mean <- rep(0, I)
Lambda_cov <- diag(10, I)

# Latent trait hyperparameters
Theta_mean <- rep(0, 2)

#############
# Stan list #
#############

stanls_md2polsi <- list(
  "P" = P,
  "I" = I,
  "C" = C,
  "D" = D,
  "Q" = Q,
  # Important transpose (array in stan are in row major order)
  "Y" = t(citems),
  "thr_mean" = Thr_mean,
  "Thr_cov" = THR_cov,
  "lambda_mean" = lambda_mean,
  "Lambda_cov" = Lambda_cov,
  "theta_mean" = Theta_mean
)

# Initialization values
lambda_init <- rnorm(I, mean = 5, sd = 1)
sum_scores <- as.matrix(citems) %*% Q
theta_init <- scale(sum_scores)

# Run MCMC chain (sample from posterior p.d.)
fit_md2polsi <- mdl_md2polsi$sample(
  data = stanls_md2polsi,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  # Mean should be below 10, since the log of it is too large
  init = function() list("lambda" = lambda_init, "theta" = theta_init)
)

###############
# Diagnostics #
###############

# Checking convergence
fit_md2polsi$cmdstan_diagnose()
fit_md2polsi$diagnostic_summary()
max(fit_md2polsi$summary()$rhat, na.rm = TRUE)
# Important: If possible, reparameterize the model

# EES & overall
print(fit_md2polsi$summary(variables = c("lambda", "mu", "Theta_cor")), n = Inf)

#########
# Draws #
#########
draws_md2polsi <- posterior::as_draws_rvars(fit_md2polsi$draws())

# Posterior distribution of latent correlation
mcmc_trace(fit_md2polsi$draws(variables = "Theta_cor[1,2]"))
mcmc_dens(fit_md2polsi$draws(variables = "Theta_cor[1,2]"))