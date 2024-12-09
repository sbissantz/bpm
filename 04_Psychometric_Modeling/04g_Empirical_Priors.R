
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

#################################
# CFA with uninformative priors #
#################################

# Compile model into executable
mdl_cfa <- cmdstan_model("./stan/4g/cfa.stan", pedantic = TRUE)

# Item intercept hyperparameters
mu_mean <- rep(0, I)
Mu_cov <- diag(1000, I)

# Item discrimination/factor loading hyperparameters
lambda_mean <- rep(0, I, I)
Lambda_cov <- diag(1000, I)

# Unique standard deviation hyperparameters
psi_rate <- rep(0.01, I)

#############
# Stan list #
#############

stanls_cfa <- list(
  "P" = P,
  "I" = I,
  "Y" = citems,
  "mu_mean" = mu_mean,
  "Mu_cov" = Mu_cov,
  "lambda_mean" = lambda_mean,
  "Lambda_cov" = Lambda_cov,
  "psi_rate" = psi_rate
)

# Fit the model to the data
fit_cfa <- mdl_cfa$sample(
  data = stanls_cfa,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  init = function() list("lambda" = rnorm(I, mean = 5, sd = 1))
)

###############
# Diagnostics #
###############

# Assess convergence: summary of all parameters
fit_cfa$cmdstan_diagnose()
fit_cfa$diagnostic_summary()

# Checking convergence
max(fit_cfa$summary()$rhat, na.rm = TRUE)

###########
# Summary #
###########

print(fit_cfa$summary(c("mu", "lambda", "psi")), n = Inf)

#########
# Draws #
#########

draws_cfa <- posterior::as_draws_rvars(fit_cfa$draws())

#############################
# CFA with empirical priors #
#############################
# 1. Empirical prior on item parameters

# Compile model into executable
mdl_cfappip <- cmdstan_model("./stan/4g/cfa_ppip.stan", pedantic = TRUE)

stanls_cfappip <- list(
  "P" = P,
  "I" = I,
  "Y" = citems,
  "mu_hypmean" = 0,
  "mu_hypsd" = 1,
  "mu_hyprate" = 0.1,
  "lambda_hypmean" = 0,
  "lambda_hypsd" = 1,
  "lambda_hyprate" = 0.1,
  "psi_hyprate" = 0.1
)

# Fit the model to the data
fit_cfappip <- mdl_cfappip$sample(
  data = stanls_cfappip,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  init = function() list("lambda" = rnorm(I, mean = 5, sd = 1))
)

###############
# Diagnostics #
###############

# Assess convergence: summary of all parameters
fit_cfappip$cmdstan_diagnose()
# Divergent transitions – reparameterize the model!
fit_cfappip$diagnostic_summary()

# Checking convergence
max(fit_cfappip$summary()$rhat, na.rm = TRUE)

###########
# Summary #
###########

print(fit_cfappip$summary(c("mu", "mu_mean", "mu_sd", "lambda", "lambda_mean", "lambda_sd", "psi", "psi_rate")), n = Inf)

#########
# Draws #
#########

draws_cfappip <- posterior::as_draws_rvars(fit_cfappip$draws())

##############
# Comparison #
##############

# Comparing intercept EAP estimates: uninformative vs. empirical prior

plot(mean(draws_cfa$mu), mean(draws_cfappip$mu),
     xlab = "Uninformative Prior", ylab = "Empirical Prior",
     main = "Comparing EAPs for Mu")

hist(mean(draws_cfa$mu) - mean(draws_cfappip$mu),
     xlab = "Mu EAP Difference", 
     main = "Uninformative Mu Prior EAP(mu) - Empirical Mu Prior EAP(mu)")

# Comparing intercept SD estimates: uninformative vs. empirical prior

plot(sd(draws_cfa$mu), sd(draws_cfappip$mu),
     xlab = "Uninformative Prior", ylab = "Empirical Prior", 
     main = "Comparing SDs for Mu")

hist(sd(draws_cfa$mu) - sd(draws_cfappip$mu),
     xlab = "Mu SD Difference",
     main = "Uninformative Mu Prior SD(mu) - Empirical Mu Prior SD(mu)")

# Comparing factor loading EAP estimates: uninformative vs. empirical prior

plot(mean(draws_cfa$lambda), mean(draws_cfappip$lambda),
     xlab = "Uninformative Prior", ylab = "Empirical Prior",
     main = "Comparing EAPs for Lambda")

hist(mean(draws_cfa$lambda) - mean(draws_cfappip$lambda),
     xlab = "Lambda EAP Difference", 
     main = "Uninformative Lambda Prior EAP(lambda) - Empirical Lambda Prior EAP(lambda)")

# Comparing factor loading SD estimates: uninformative vs. empirical prior

plot(sd(draws_cfa$lambda), sd(draws_cfappip$lambda),
     xlab = "Uninformative Prior", ylab = "Empirical Prior", 
     main = "Comparing SDs for Lambda")

hist(sd(draws_cfa$lambda) - sd(draws_cfappip$lambda),
     xlab = "Lambda SD Difference", 
     main = "Uninformative Lambda Prior SD(lambda) - Empirical Lambda Prior SD(lambda)")

# Comparing unique EAP estimates: uninformative vs. empirical prior
plot(mean(draws_cfa$psi), mean(draws_cfappip$psi),
     xlab = "Uninformative Prior", ylab = "Empirical Prior", 
     main = "Comparing EAPs for Psi")

hist(mean(draws_cfa$psi) - mean(draws_cfappip$psi),
     xlab = "Psi EAP Difference",
     main = "Uninformative Psi Prior EAP(psi) - Empirical Psi Prior EAP(psi)")



# empirical prior for theta ===========================================================

modelCFA3_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  matrix[nObs, nItems] Y;            // item responses in a matrix

  real meanLambdaMean;
  real<lower=0> meanLambdaSD;
  real<lower=0> sdLambdaRate;
  
  real meanMuMean;
  real<lower=0> meanMuSD;
  real<lower=0> sdMuRate;
  
  real<lower=0> ratePsiRate;
  
  real meanThetaMean;
  real<lower=0> meanThetaSD;
  real<lower=0> sdThetaRate;
}

parameters {
  vector[nObs] theta;                
  real meanTheta;
  real<lower=0> sdTheta;
    
  vector[nItems] lambda;
  real meanLambda;
  real<lower=0> sdLambda;
  
  vector[nItems] mu;
  real meanMu;
  real<lower=0> sdMu;
  
  vector<lower=0>[nItems] psi;
  real<lower=0> psiRate;
}

model {
  
  meanLambda ~ normal(meanLambdaMean, meanLambdaSD);
  sdLambda ~ exponential(sdLambdaRate);
  lambda ~ normal(meanLambda, sdLambda);
  
  meanMu ~ normal(meanMuMean, meanMuSD);
  sdMu ~ exponential(sdMuRate);
  mu ~ normal(meanMu, sdMu); 
  
  psiRate ~ exponential(ratePsiRate);
  psi ~ exponential(psiRate);            
  
  meanTheta ~ normal(meanThetaMean, meanThetaSD);
  sdTheta ~ exponential(sdThetaRate);
  theta ~ normal(meanTheta, sdTheta);                  
  
  for (item in 1:nItems){
    Y[,item] ~ normal(mu[item] + lambda[item]*theta, psi[item]);
  }
  
}

"
modelCFA3_stan = cmdstan_model(stan_file = write_stan_file(modelCFA3_syntax))

modelCFA3_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = conspiracyItems, 
  meanLambdaMean = 0,
  meanLambdaSD = 1,
  sdLambdaRate = .1,
  meanMuMean = 0,
  meanMuSD = 1,
  sdMuRate = .1,
  ratePsiRate = .1,
  meanThetaMean = 0,
  meanThetaSD = 1,
  sdThetaRate = .1
)


modelCFA3_samples = modelCFA3_stan$sample(
  data = modelCFA3_data,
  seed = 191120223,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelCFA3_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(
  modelCFA3_samples$summary(
    variables = c("meanTheta", "sdTheta", "mu", "meanMu", "sdMu", "lambda", "meanLambda", "sdLambda", "psi", "psiRate")
  ), 
  n=Inf
)

mcmc_trace(modelCFA3_samples$draws(variables = c("meanTheta", "sdTheta")))

# model4: empirical theta prior, fixed prior on item parameters
modelCFA4_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  matrix[nObs, nItems] Y;            // item responses in a matrix

  vector[nItems] meanMu;
  matrix[nItems, nItems] covMu;      // prior covariance matrix for coefficients
  
  vector[nItems] meanLambda;         // prior mean vector for coefficients
  matrix[nItems, nItems] covLambda;  // prior covariance matrix for coefficients
  
  vector[nItems] psiRate;            // prior rate parameter for unique standard deviations
  
  real meanThetaMean;
  real<lower=0> meanThetaSD;
  real<lower=0> sdThetaRate;
  
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  real meanTheta;
  real<lower=0> sdTheta;
  vector[nItems] mu;                 // the item intercepts (one for each item)
  vector[nItems] lambda;             // the factor loadings/item discriminations (one for each item)
  vector<lower=0>[nItems] psi;       // the unique standard deviations (one for each item)   
}

model {
  
  lambda ~ multi_normal(meanLambda, covLambda); // Prior for item discrimination/factor loadings
  mu ~ multi_normal(meanMu, covMu);             // Prior for item intercepts
  psi ~ exponential(psiRate);                   // Prior for unique standard deviations
  
  meanTheta ~ normal(meanThetaMean, meanThetaSD);
  sdTheta ~ exponential(sdThetaRate);
  theta ~ normal(meanTheta, sdTheta);                  
  
  for (item in 1:nItems){
    Y[,item] ~ normal(mu[item] + lambda[item]*theta, psi[item]);
  }
  
}

"

modelCFA4_stan = cmdstan_model(stan_file = write_stan_file(modelCFA4_syntax))


# item intercept hyperparameters
muMeanHyperParameter = 0
muMeanVecHP = rep(muMeanHyperParameter, nItems)

muVarianceHyperParameter = 1
muCovarianceMatrixHP = diag(x = muVarianceHyperParameter, nrow = nItems)

# item discrimination/factor loading hyperparameters
lambdaMeanHyperParameter = 0
lambdaMeanVecHP = rep(lambdaMeanHyperParameter, nItems)

lambdaVarianceHyperParameter = 1
lambdaCovarianceMatrixHP = diag(x = lambdaVarianceHyperParameter, nrow = nItems)

# unique standard deviation hyperparameters
psiRateHyperParameter = .01
psiRateVecHP = rep(psiRateHyperParameter, nItems)


modelCFA4_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = conspiracyItems, 
  meanMu = muMeanVecHP,
  covMu = muCovarianceMatrixHP,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP,
  psiRate = psiRateVecHP,
  meanThetaMean = 0,
  meanThetaSD = 1,
  sdThetaRate = 1
)

modelCFA4_samples = modelCFA4_stan$sample(
  data = modelCFA4_data,
  seed = 191120224,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelCFA4_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelCFA4_samples$summary(variables = c("meanTheta", "sdTheta", "mu", "lambda", "psi")) ,n=Inf)

save.image(file = "lecture04g.RData")


