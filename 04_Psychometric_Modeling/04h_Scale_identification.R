
#########
# Setup #
#########

library(cmdstanr)
library(bayesplot)
library(ggplot2)
library(posterior)

########
# Data #
########

# Model the belief in conspiracy theories,
# Assuming a binomial p.d. for the data
conspiracy_data <- read.csv("./data/conspiracies.csv")

# Only using the first 10 items
# Positive values mean resemble agreement
citems <- conspiracy_data[, 1 : 10]

# Number of items 
I <- ncol(citems)

# Number of respondents
P <- nrow(citems)

# Number of categories 
# Note: Successive integers from 1 to highest 
C <- max(citems)

# Item means
# Note: the mean is the proportion of respondents who agreed with the item
colMeans(citems)

##############################
# 2 POL SI (ordered-logit)   #
# (Slope-Intercept Form)     # 
# aka. Graded Response Model #
##############################

# Compile model
mdl_2polsi <- cmdstan_model("./stan/4d/2pol_si.stan", pedantic = TRUE)

# Item threshold hyperparameters
Thr_mean <- replicate(C - 1, rep(0, I)) # 10 x 4
THR_cov <- array(0, dim = c(10, 4, 4)) # 10 x 4 x 4
for(d in seq_len(I)) {
  THR_cov[d , ,] <- diag(1000, C - 1)
}

# Item discrimination/factor loading hyperparameters
lambda_mean <- rep(0, I)
Lambda_cov <- diag(1000, I)

stanls_2polsi <- list(
  "P" = P,
  "I" = I,
  "C" = C,
  # Important transpose (array in stan are in row major order)
  "Y" = t(citems),
  "thr_mean" = Thr_mean,
  "Thr_cov" = THR_cov,
  "lambda_mean" = lambda_mean,
  "Lambda_cov" = Lambda_cov 
)

# Multiple standardized sum scores to start theta in the correct location
sum_scores <- as.matrix(citems) %*% matrix(1, I, 1)
theta_init <- as.numeric(scale(sum_scores))

# Run MCMC chain (sample from posterior p.d.)
fit_2polsi <- mdl_2polsi$sample(
  data = stanls_2polsi,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  # Mean should be below 10, since the log of it is too large
  init = function() list("lambda" = rnorm(I, mean = 10, sd = 1), "theta" = theta_init))

#########
# Draws #
#########

draws_2polsi <- posterior::as_draws_rvars(fit_2polsi$draws())

##############################
# 2 POL SI (ordered-logit)   #
# aka. Graded Response Model #
##############################
# Scale Identification with marker items 
# Estimate the factor variance – with strong identification (data likelihood
# and posterior p.d.); that is, identification with not just prior information
# Note: Usually, this is only taught in SEM contexts, but here: GRM!

# Compile model
mdl_2polsim <- cmdstan_model("./stan/4h/2pol_si_marker.stan", pedantic = TRUE)

# Item discrimination/factor loading hyperparameters
# Important: I-1!
lambda_mean <- rep(0, I - 1)
Lambda_cov <- diag(1000, I - 1)

#############
# Stan list #
#############

stanls_2polsim <- list(
  "P" = P,
  "I" = I,
  "C" = C,
  # Important transpose (array in stan are in row major order)
  "Y" = t(citems),
  "thr_mean" = Thr_mean,
  "Thr_cov" = THR_cov,
  "lambda_mean" = lambda_mean,
  "Lambda_cov" = Lambda_cov,
  "theta_hypermean" = 0,
  "theta_hypersd" = 2
)

# Multiple standardized sum scores to start theta in the correct location
sum_scores <- as.matrix(citems) %*% matrix(1, I, 1)
theta_init <- as.numeric(scale(sum_scores))

# Run MCMC chain (sample from posterior p.d.)
fit_2polsim <- mdl_2polsim$sample(
  data = stanls_2polsim,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  # Mean should be below 10, since the log of it is too large
  init = function() list("lambda" = rnorm(I, mean = 10, sd = 1), "theta" = theta_init))

##############
# Diagnostic #
##############

# Checking convergence
fit_2polsim$cmdstan_diagnose()
fit_2polsim$diagnostic_summary()
max(fit_2polsim$summary()$rhat, na.rm = TRUE)

# EES & overall
print(fit_2polsim$summary(), n = Inf)

# Trace and density plots
mcmc_trace(fit_2polsim$draws(variables = c("theta_sd")))
mcmc_dens(x = fit_2polsim$draws(variables = c("theta_sd")))

###########
# Summary #
###########

# Loading parameter
fit_2polsim$summary("mu")

# Loading parameter
fit_2polsim$summary("lambda")
# Note: This is our rate of change in expectation of y for a one-unit change in 
# theta. But now this 1 unit is not 1 SD but the factor standard deviation. 

# Estimated factor standard deviation 
fit_2polsim$summary("theta_sd")

#########
# Draws #
#########

draws_2polsim <- posterior::as_draws_rvars(fit_2polsim$draws())

##############
# Comparison #
##############

# Theta EAP estimates

plot(density(mean(draws_2polsim$theta)),
     ylim = c(0, max(density(mean(draws_2polsim$theta))$y)),
     col = 2, lwd = 3, main = "Comparing Theta EAP Densities")
lines(density(mean(draws_2polsi$theta)), col = 3, lwd = 3)
legend("topright", legend = c("Marker Item", "Standardized Factor"), 
col = c(2, 3), lwd = c(2, 3), lty = c(1, 1))
legend("topright", legend = c("Marker Item", "Standardized Factor"), 
col = c(2, 3), lwd = c(2, 3), lty = c(1, 1))

plot(mean(draws_2polsim$theta), mean(draws_2polsi$theta), 
     xlab = "Marker Item", ylab = "Standardized Factor", main = "Comparing Theta EAP Estimates")

# Theta SD estimates

# Density comparison
plot(density(sd(draws_2polsim$theta)),
     ylim = c(0, max(density(sd(draws_2polsim$theta))$y)),
     col = 2, lwd = 3, main = "Comparing Theta SD Densities")
lines(density(sd(draws_2polsi$theta)), col = 3, lwd = 3)
legend("topright", legend = c("Marker Item", "Standardized Factor"), 
col = c(2, 3), lwd = c(2, 3), lty = c(1, 1))

# Scatterplot
plot(sd(draws_2polsim$theta), sd(draws_2polsi$theta), 
     xlab = "Marker Item", ylab = "Standardized Factor", main = "Comparing Theta SD Estimates")

# Lambda EAP estimates

# Density comparison
plot(density(mean(draws_2polsim$lambda)),
     ylim = c(0, max(density(mean(draws_2polsim$lambda))$y)),
     col = 2, lwd = 3, main = "Comparing Lambda EAP Estimates")
lines(density(mean(draws_2polsi$lambda)), col = 3, lwd = 3)
legend("topright", legend = c("Marker Item", "Standardized Factor"), 
col = c(2, 3), lwd = c(2, 3), lty = c(1, 1))

# Scatterplot
plot(mean(draws_2polsim$lambda), mean(draws_2polsi$lambda), 
     xlab = "Marker Item", ylab = "Standardized Factor", main = "Comparing Lambda EAP Estimates")
# The marker item shifts where the following lambda values go

# Lambda SD estimates

# Density comparison
plot(density(sd(draws_2polsim$lambda)),
     ylim = c(0, max(density(sd(draws_2polsim$lambda))$y)),
     col = 2, lwd = 3, main = "Comparing Theta SD Estimates")
lines(density(sd(draws_2polsi$lambda)), col = 3, lwd = 3)
legend("topright", legend = c("Marker Item", "Standardized Factor"), 
col = c(2, 3), lwd = c(2, 3), lty = c(1, 1))

# Scatterplot
plot(sd(draws_2polsim$lambda), sd(draws_2polsi$lambda), 
     xlab = "Marker Item", ylab = "Standardized Factor", main = "Comparing Theta EAD Estimates")
# The marker item shifts where the following lambda values go

# todo todo todo!

# Multidimesnional GRM + Factor Variance: Strong identification (model/data likelihood and posterior) ===========================


modelMultidimensionalGRM_markerItem_syntax = "
data {
  
  // data specifications  =============================================================
  int<lower=0> nObs;                            // number of observations
  int<lower=0> nItems;                          // number of items
  int<lower=0> maxCategory;                     // number of categories for each item
  
  // input data  =============================================================
  array[nItems, nObs] int<lower=1, upper=5>  Y; // item responses in an array

  // loading specifications  =============================================================
  int<lower=1> nFactors;                                       // number of loadings in the model
  array[nItems, nFactors] int<lower=0, upper=1> Qmatrix;
  
  // prior specifications =============================================================
  array[nItems] vector[maxCategory-1] meanThr;                // prior mean vector for intercept parameters
  array[nItems] matrix[maxCategory-1, maxCategory-1] covThr;  // prior covariance matrix for intercept parameters
  
  vector[nItems-nFactors] meanLambda;         // prior mean vector for discrimination parameters
  matrix[nItems-nFactors, nItems-nFactors] covLambda;  // prior covariance matrix for discrimination parameters
  
  vector[nFactors] meanTheta; 
  vector[nFactors] sdThetaLocation;
  vector[nFactors] sdThetaScale;
}

transformed data{
  int<lower=0> nLoadings = 0;                                      // number of loadings in model
  array[nFactors] int<lower=0> markerItem = rep_array(0, nFactors);
  
  for (factor in 1:nFactors){
    nLoadings = nLoadings + sum(Qmatrix[1:nItems, factor]);
  }

  array[nLoadings, 4] int loadingLocation;                     // the row/column positions of each loading, plus marker switch
  
  int loadingNum=1;
  int lambdaNum=1;
  for (item in 1:nItems){
    for (factor in 1:nFactors){       
      if (Qmatrix[item, factor] == 1){
        loadingLocation[loadingNum, 1] = item;
        loadingLocation[loadingNum, 2] = factor;
        if (markerItem[factor] == 0){
          loadingLocation[loadingNum, 3] = 1;     // ==1 if marker item, ==0 otherwise
          loadingLocation[loadingNum, 4] = 0;     // ==0 if not one of estimated lambdas
          markerItem[factor] = item;
        } else {
          loadingLocation[loadingNum, 3] = 0;
          loadingLocation[loadingNum, 4] = lambdaNum;
          lambdaNum = lambdaNum + 1;
        }
        loadingNum = loadingNum + 1;
      }
    }
  }


}

parameters {
  array[nObs] vector[nFactors] theta;                // the latent variables (one for each person)
  array[nItems] ordered[maxCategory-1] thr; // the item thresholds (one for each item category minus one)
  vector[nLoadings-nFactors] initLambda;             // the factor loadings/item discriminations (one for each item)
  
  cholesky_factor_corr[nFactors] thetaCorrL;
  vector<lower=0>[nFactors] thetaSD;
}

transformed parameters{
  matrix[nItems, nFactors] lambdaMatrix = rep_matrix(0.0, nItems, nFactors);
  matrix[nObs, nFactors] thetaMatrix;
  
  // build matrix for lambdas to multiply theta matrix
  
  for (loading in 1:nLoadings){  
    if (loadingLocation[loading,3] == 1){
      lambdaMatrix[loadingLocation[loading,1], loadingLocation[loading,2]] = 1.0;
    } else {
      lambdaMatrix[loadingLocation[loading,1], loadingLocation[loading,2]] = initLambda[loadingLocation[loading,4]];
    }
  }
  
  for (factor in 1:nFactors){
    thetaMatrix[,factor] = to_vector(theta[,factor]);
  }
  
}

model {
  
  matrix[nFactors, nFactors] thetaCovL;
  initLambda ~ multi_normal(meanLambda, covLambda); 
  
  thetaCorrL ~ lkj_corr_cholesky(1.0);
  thetaSD ~ lognormal(sdThetaLocation,sdThetaScale);
  
  thetaCovL = diag_pre_multiply(thetaSD, thetaCorrL);
  theta ~ multi_normal_cholesky(meanTheta, thetaCovL);    
  
  
  for (item in 1:nItems){
    thr[item] ~ multi_normal(meanThr[item], covThr[item]);            
    Y[item] ~ ordered_logistic(thetaMatrix*lambdaMatrix[item,1:nFactors]', thr[item]);
  }
  
  
}

generated quantities{ 
  array[nItems] vector[maxCategory-1] mu;
  corr_matrix[nFactors] thetaCorr;
  cholesky_factor_cov[nFactors] thetaCov_pre;
  cov_matrix[nFactors] thetaCov; 
  
  for (item in 1:nItems){
    mu[item] = -1*thr[item];
  }
  
  thetaCorr = multiply_lower_tri_self_transpose(thetaCorrL);
  thetaCov_pre = diag_pre_multiply(thetaSD, thetaCorrL);
  thetaCov = multiply_lower_tri_self_transpose(thetaCov_pre);
}


"

modelMultidimensionalGRM_markerItem_stan = cmdstan_model(stan_file = write_stan_file(modelMultidimensionalGRM_markerItem_syntax))


# Build a Q-Matrix ===========================================================================

Qmatrix = matrix(data = 0, nrow = ncol(conspiracyItems), ncol = 2)
colnames(Qmatrix) = c("Gov", "NonGov")
rownames(Qmatrix) = paste0("item", 1:ncol(conspiracyItems))
Qmatrix[1,2] = 1
Qmatrix[2,1] = 1
Qmatrix[3,2] = 1
Qmatrix[4,2] = 1
Qmatrix[5,1] = 1
Qmatrix[6,2] = 1
Qmatrix[7,1] = 1
Qmatrix[8,1] = 1
Qmatrix[9,1] = 1
Qmatrix[10,2] = 1

Qmatrix


# Data needs: successive integers from 1 to highest number (recode if not consistent)
maxCategory = 5

# data dimensions
nObs = nrow(conspiracyItems)
nItems = ncol(conspiracyItems)
nFactors = ncol(Qmatrix)

# item threshold hyperparameters
thrMeanHyperParameter = 0
thrMeanVecHP = rep(thrMeanHyperParameter, maxCategory-1)
thrMeanMatrix = NULL
for (item in 1:nItems){
  thrMeanMatrix = rbind(thrMeanMatrix, thrMeanVecHP)
}

thrVarianceHyperParameter = 1000
thrCovarianceMatrixHP = diag(x = thrVarianceHyperParameter, nrow = maxCategory-1)
thrCovArray = array(data = 0, dim = c(nItems, maxCategory-1, maxCategory-1))
for (item in 1:nItems){
  thrCovArray[item, , ] = diag(x = thrVarianceHyperParameter, nrow = maxCategory-1)
}

# item discrimination/factor loading hyperparameters
lambdaMeanHyperParameter = 0
lambdaMeanVecHP = rep(lambdaMeanHyperParameter, nItems-nFactors)

lambdaVarianceHyperParameter = 1000
lambdaCovarianceMatrixHP = diag(x = lambdaVarianceHyperParameter, nrow = nItems-nFactors)

# theta hyperparameters
thetaMean = rep(0, nFactors)
sdThetaLocation = rep(0, nFactors)
sdThetaScale = rep(.5, nFactors)

modelMultidimensionalGRM_markerItem_data = list(
  nObs = nObs,
  nItems = nItems,
  maxCategory = maxCategory,
  Y = t(conspiracyItems), 
  nFactors = nFactors,
  Qmatrix = Qmatrix,
  meanThr = thrMeanMatrix,
  covThr = thrCovArray,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP,
  meanTheta = thetaMean,
  sdThetaLocation = sdThetaLocation,
  sdThetaScale = sdThetaScale
)

modelMultidimensionalGRM_markerItem_samples = modelMultidimensionalGRM_markerItem_stan$sample(
  data = modelMultidimensionalGRM_markerItem_data,
  seed = 201120224,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  thin = 2,
  init = function() list(lambda=rnorm(nItems, mean=10, sd=1))
)

 # checking convergence
max(modelMultidimensionalGRM_markerItem_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelMultidimensionalGRM_markerItem_samples$summary(variables = c("thetaSD", "thetaCov", "thetaCorr", "lambdaMatrix", "mu")) ,n=Inf)

save.image(file = "lecture04h.RData")
