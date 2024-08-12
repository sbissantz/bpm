#################################################
# Efficient Stan Code and Generated Quantities #
#################################################

library(cmdstanr)
# bayesplot: for plotting posterior distributions
library(bayesplot)
# HDInterval: for constructing Highest Density Posterior Intervals
library(HDInterval)
library(ggplot2)

########
# data #
########

DietData <- read.csv(file = "./data/DietData.csv")

# centering variables
DietData$Height60IN <- DietData$HeightIN - 60

# dummy variable for group 2
(group2 <- rep(0, nrow(DietData)))
(pat <- which(DietData$DietGroup == 2))
group2[pat] <- 1

# dummy variable for group 3
group3 <- rep(0, nrow(DietData))
pat <- which(DietData$DietGroup == 3)
group3[pat] <- 1

# interaction terms
heightXgroup2 <- DietData$Height60IN * group2
heightXgroup3 <- DietData$Height60IN * group3

# adding prior to betas
fml <- "

data {
  int<lower=0> N;
  vector[N] weightLB;
  vector[N] height60IN;
  vector[N] group2;
  vector[N] group3;
  vector[N] heightXgroup2;
  vector[N] heightXgroup3;
}


parameters {
  real beta0;
  real betaHeight;
  real betaGroup2;
  real betaGroup3;
  real betaHxG2;
  real betaHxG3;
  real<lower=0> sigma;
}


model {
  beta0 ~ normal(0,1000);
  betaHeight ~ normal(0,1000);
  betaGroup2 ~ normal(0,1000);
  betaGroup3 ~ normal(0,1000);
  betaHxG2 ~ normal(0,1000);
  betaHxG3 ~ normal(0,1000);
  
  sigma ~ exponential(.1); // prior for sigma
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, sigma);
}

"

# compile model
mdl04 <- cmdstan_model(stan_file = write_stan_file(fml), pedantic = TRUE)

# build r list for stan
stanls <- list(
  N = nrow(DietData),
  weightLB = DietData$WeightLB,
  height60IN = DietData$Height60IN,
  group2 = group2,
  group3 = group3,
  heightXgroup2 = heightXgroup2,
  heightXgroup3 = heightXgroup3
)

# run MCMC chain (sample from posterior p.d.)
fit04 <- model04_Stan$sample(
  data = stanls,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

# assess convergence: summary of all parameters
fit04$summary()

###############
# Matrix form #
###############
# making it easier to supply input into Stan
# important: no need to recomplie the model with new data

# Model equation
# y = f(X) = X * beta + alpha + epsilon

# adding multivariate prior p.d. to betas
fml <- "

data {
  int<lower=0> N;         // number of observations
  int<lower=0> P;         // number of predictors (plus column for intercept)
  matrix[N, P] X;         // model.matrix() from R 
  vector[N] y;            // outcome
  
  vector[P] meanBeta;       // prior mean vector for coefficients
  matrix[P, P] covBeta; // prior covariance matrix for coefficients
  
  real sigmaRate;         // prior rate parameter for residual sd
}


parameters {
  vector[P] beta;         // vector of coefficients for Beta
  real<lower=0> sigma;    // residual standard deviation
}


model {
  beta ~ multi_normal(meanBeta, covBeta); // prior for coefficients
  sigma ~ exponential(sigmaRate);         // prior for sigma
  y ~ normal(X*beta, sigma);              // linear model
}

"

# start with model formula
mdl05_fml <- formula(WeightLB ~ Height60IN + factor(DietGroup) + 
Height60IN:factor(DietGroup), data = DietData)

# grab model matrix
mdl05_predictorMatrix <- model.matrix(mdl05_fml, data = DietData)
dim(model05_predictorMatrix)

# find details of model matrix
N <- nrow(mdl05_predictorMatrix)
P <- ncol(mdl05_predictorMatrix)

# build matrices of hyper parameters (for priors)
(meanBeta <- rep(0, P))
(covBeta <- diag(x = 10000, nrow = P, ncol = P))
(sigmaRate <- .1)

# build r list for stan
stanls <- list(
  N = N,
  P = P,
  X = model05_predictorMatrix,
  y = DietData$WeightLB,
  meanBeta = meanBeta,
  covBeta = covBeta,
  sigmaRate = sigmaRate
)

# compile model -- this method is for stan code as a string
mdl05 <- cmdstan_model(stan_file = write_stan_file(fml), pedantic = TRUE)

# run MCMC chain (sample from posterior p.d.)
fit05 <- mdl05$sample(
  data = model05_data,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000
)

# assess convergence: summary of all parameters
fit05$summary()
fit05$cmdstan_diagnose()
fit05$print()
fit05$diagnostic_summary()

# maximum R-hat
max(fit05$summary()$rhat)

######################
# diet group 2 slope #
######################

slopeG2 <- fit05$draws("beta[2]") + fit05$draws("beta[4]")
summary(slopeG2)

# posterior histograms
mcmc_hist(slopeG2)

# posterior densities
mcmc_dens(slopeG2)

# Forming Diet Group 2 via Stan Generated Quantites (scalar version) ==========

# adding prior to betas
fml <- "

data {
  int<lower=0> N;
  vector[N] weightLB;
  vector[N] height60IN;
  vector[N] group2;
  vector[N] group3;
  vector[N] heightXgroup2;
  vector[N] heightXgroup3;
}

parameters {
  real beta0;
  real betaHeight;
  real betaGroup2;
  real betaGroup3;
  real betaHxG2;
  real betaHxG3;
  real<lower=0> sigma;
}

model {
  beta0 ~ normal(0,1);
  betaHeight ~ normal(0,1000);
  betaGroup2 ~ normal(0,1000);
  betaGroup3 ~ normal(0,1000);
  betaHxG2 ~ normal(0,1000);
  betaHxG3 ~ normal(0,1000);
  
  sigma ~ exponential(.1); // prior for sigma
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, sigma);
}

generated quantities {
  real slopeG2;
  slopeG2 = betaHeight + betaHxG2;
}

"

# compile model -- this method is for stan code as a string
mdl04b <- cmdstan_model(stan_file = write_stan_file(fml))

# build r list for stan
stanls <- list(
  N = nrow(DietData),
  weightLB = DietData$WeightLB,
  height60IN = DietData$Height60IN,
  group2 = group2,
  group3 = group3,
  heightXgroup2 = heightXgroup2,
  heightXgroup3 = heightXgroup3
)

fit04b = mdl04b$sample(
  data = mode04b_StanData,
  seed = 190920221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

# assess convergence: summary of all parameters
fit04b$summary()

##########################
# Matrices and Contrasts #
##########################

# adding prior to betas
fml <- "

data {
  int<lower=0> N;         // number of observations
  int<lower=0> P;         // number of predictors (plus column for intercept)
  matrix[N, P] X;         // model.matrix() from R 
  vector[N] y;            // outcome
  
  vector[P] meanBeta;     // prior mean vector for coefficients
  matrix[P, P] covBeta;   // prior covariance matrix for coefficients
  
  real sigmaRate;         // prior rate parameter for residual standard deviation
  
  int<lower=0> nContrasts; 
  matrix[nContrasts,P] contrastMatrix;   // contrast matrix for additional effects
}

parameters {
  vector[P] beta;         // vector of coefficients for Beta
  real<lower=0> sigma;    // residual standard deviation
}

model {
  beta ~ multi_normal(meanBeta, covBeta); // prior for coefficients
  sigma ~ exponential(sigmaRate);         // prior for sigma
  y ~ normal(X*beta, sigma);              // linear model
}

generated quantities {
  vector[nContrasts] contrasts;
  contrasts = contrastMatrix*beta;
}

"

nContrasts <- 2
(contrastMatrix <- matrix(data = 0, nrow = nContrasts, ncol = P))
contrastMatrix[1,2] = contrastMatrix[1,5] = 1 # for slope for group=2
contrastMatrix[2,1] = contrastMatrix[2,3] = 1 # for intercept for group=2
contrastMatrix

# build Stan data from model matrix
stanls = list(
  N = N,
  P = P,
  X = model05_predictorMatrix,
  y = DietData$WeightLB,
  meanBeta = meanBeta,
  covBeta = covBeta,
  sigmaRate = sigmaRate,
  contrastMatrix = contrastMatrix,
  nContrasts = nContrasts
)

# compile model -- this method is for stan code as a string
mdl05b <- cmdstan_model(stan_file = write_stan_file(fml), pedantic = TRUE)

fit05b <- mdl05b$sample(
  data = stanls,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000
)

# assess convergence: summary of all parameters
fit05b$summary()

# assess convergence: summary of all parameters
max(fit05b$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(fit05b$draws("contrasts[1]"))

# posterior histograms
mcmc_hist(fit05b$draws("contrasts[1]"))

# posterior densities
mcmc_dens(fit05b$draws("contrasts[1]"))

################################
# Matrices, Contrasts, and R^2 #
################################

# adding prior to betas
fml <- "

data {
  int<lower=0> N;         // number of observations
  int<lower=0> P;         // number of predictors (plus column for intercept)
  matrix[N, P] X;         // model.matrix() from R 
  vector[N] y;            // outcome
  
  vector[P] meanBeta;     // prior mean vector for coefficients
  matrix[P, P] covBeta;   // prior covariance matrix for coefficients
  
  real sigmaRate;         // prior rate parameter for residual standard deviation
  
  int<lower=0> nContrasts; 
  matrix[nContrasts,P] contrastMatrix;   // contrast matrix for additional effects
}


parameters {
  vector[P] beta;         // vector of coefficients for Beta
  real<lower=0> sigma;    // residual standard deviation
}

model {
  beta ~ multi_normal(meanBeta, covBeta); // prior for coefficients
  sigma ~ exponential(sigmaRate);         // prior for sigma
  y ~ normal(X*beta, sigma);              // linear model
}

generated quantities {
  vector[nContrasts] contrasts;
  contrasts = contrastMatrix*beta;
  
  real rss;
  real totalrss;
  {
    vector[N] pred;
    pred = X*beta;
    rss = dot_self(y-pred); // Dot product of a vector with itself (scalar)
    totalrss = dot_self(y-mean(y));
  }
  
  real R2;
  R2 = 1-rss/totalrss;
}

"

nContrasts <- 2 
contrastMatrix <- matrix(data = 0, nrow = nContrasts, ncol = P)
contrastMatrix[1,2] = contrastMatrix[1,5] = 1 # for slope for group=2
contrastMatrix[2,1] = contrastMatrix[2,3] = 1 # for intercept for group=2

# build Stan data from model matrix
stanls = list(
  N = N,
  P = P,
  X = model05_predictorMatrix,
  y = DietData$WeightLB,
  meanBeta = meanBeta,
  covBeta = covBeta,
  sigmaRate = sigmaRate,
  contrastMatrix = contrastMatrix,
  nContrasts = nContrasts
)

# compile model -- this method is for stan code as a string
mdl05c <- cmdstan_model(stan_file = write_stan_file(fml), pedantic = TRUE)

# run MCMC chain (sample from posterior p.d.)
fit05c = mdl05c$sample(
  data = stanls,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000
)

# assess convergence: summary of all parameters
fit05c$cmdstan_diagnose()
fit05c$print()
fit05c$diagnostic_summary()

# visualize posterior timeseries
mcmc_trace(fit05c$draws("R2"))

# posterior histograms
mcmc_hist(fit05c$draws("R2"))

# posterior densities
mcmc_dens(fit05c$draws("R2"))
