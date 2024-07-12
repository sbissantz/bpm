# #################################
# Example_Linear_Models_with_MCMC #
# #################################

library(ggplot2)
library(cmdstanr)
library(bayesplot)
library(modeest)

########
# data #
########

# outcome: Weight in pounds
DietData <- read.csv(file = "DietData.csv")

# important that we know what 0 is in our interaction
# center predictor variable
# gives E(Y | X = 60) NOT 0 anymore
DietData$Height60IN <- DietData$HeightIN-60

#################
# visualization #
#################

ggplot(data = DietData, aes(x = WeightLB)) + 
  geom_histogram(aes(y = ..density..), position = "identity", binwidth = 10) + 
  geom_density(alpha=.2) 

ggplot(data = DietData, aes(x = WeightLB, color = factor(DietGroup), fill = factor(DietGroup))) + 
  geom_histogram(aes(y = ..density..), position = "identity", binwidth = 10) + 
  geom_density(alpha=.2) 

ggplot(data = DietData, aes(x = HeightIN, y = WeightLB, shape = factor(DietGroup), color = factor(DietGroup))) +
  geom_smooth(method = "lm", se = FALSE) + geom_point()

#################
# linear model  #
#################

# Asm  y = f(X) + e = beta0 + beta1*X + e
# where e ~ N(0, sigma_e^2)

############################
# Ordinary least squares  #
########################### 

# full analysis model suggested by data: =======================================
full_fml <- WeightLB ~ Height60IN + factor(DietGroup) + 
  Height60IN:factor(DietGroup)
full_mdl <- lm(formula = full_fml, data = DietData)

# examining assumptions and leverage of fit
plot(full_mdl)

# looking at ANOVA table
anova(full_mdl)

# looking at parameter summary
summary(full_mdl)

# building Stan code for same model -- initially wihtout priors

fml_mdl01 <- "

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
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, sigma);
}

"

#############
# stan list #
#############

group2 <- rep(0, nrow(DietData))
group2[which(DietData$DietGroup == 2)] <- 1

group3 <- rep(0, nrow(DietData))
group3[which(DietData$DietGroup == 3)]<- 1

heightXgroup2 <- DietData$Height60IN * group2
heightXgroup3 <- DietData$Height60IN * group3

# building Stan data into R list
stanls_mdl01 <- list(
  N = nrow(DietData),
  weightLB = DietData$WeightLB,
  height60IN = DietData$Height60IN,
  group2 = group2,
  group3 = group3,
  heightXgroup2 = heightXgroup2,
  heightXgroup3 = heightXgroup3
)

# compile model -- this method is for stan code as a string
mdl01 <- cmdstan_model(stan_file = write_stan_file(fml_mdl01))

# run MCMC chain (sample from posterior p.d.)
fit01 <- mdl01$sample(
  data = stanls_mdl01,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 10000,
  iter_sampling = 10000
)

###############
# diagnostics #
###############

# assess convergence: summary of all parameters
fit01$print()
fit01$cmdstan_diagnose()
fit01$diagnostic_summary()

# maximum R-hat
max(fit01$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(fit01$draws())

# posterior histograms
mcmc_hist(fit01$draws())

# posterior densities
mcmc_dens(fit01$draws())

# next: compare model results with results from lm()

stanSummary01 <- fit01$summary()
lsSummary01 <- summary(full_mdl)

################################################################################
# comparison of fixed effects
cbind(lsSummary01$coefficients[, 1:2], stanSummary01[2:7, c(2, 4)])

# what do you notice being different?
# -> posterior standard deviations are always bigger than standard errors!
################################################################################

################################################################################
# comparison of residual standard deviation
cbind(lsSummary01$sigma, stanSummary01[8, c(2:4)])

# what do you notice being different?
# -> residual standard deviation versus is bigger than residual standard errors!

# Why?

# Integrate full p.d. of sigmma 
mcmc_dens(fit01$draws(variables = "sigma"))

# ...instead of just looking at the mode (point estimate)

# calculating mode of posterior for sigma
mlv(fit01$draws("sigma"), method = "meanshift")
################################################################################

# building model to mirror ls model
fml_mdl02 <- "

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
}


model {
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    // IMPORTANT residual standard error from lm model (7.949437)
    betaHxG3 * heightXgroup3, 7.949437); 
}

"

# 7.949437 is the Residual standard error from the lm model 
summary(full_mdl)$sigma

# compile model -- this method is for stan code as a string
mdl02 <- cmdstan_model(stan_file = write_stan_file(fml_mdl02))

# run MCMC chain (sample from posterior p.d.)
fit02 <- mdl02$sample(
  data = stanls_mdl01,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1e3,
  iter_sampling = 1e5
)

###############
# diagnositcs #
###############

fit02$cmdstan_diagnose()
fit02$print()
fit02$diagnostic_summary()

# maximum R-hat
max(fit02$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(fit02$draws())

# posterior histograms
mcmc_hist(fit02$draws())

# posterior densities
mcmc_dens(fir02$draws())

# comparison of results
stanSummary02 <- fit02$summary()
lsSummary02 <- summary(full_mdl)

# comparison of fixed effects
cbind(lsSummary02$coefficients[,1:2], stanSummary02[2:7,c(2,4)])

################################################################################
# what do you notice being different?
# -> after fixing the resudual standard deviation, the posterior p.d. standard 
# deviations is identical to the residual standard error 
#
# IMPORT: Bayes resolves the ASM of asymptotic convergence! 
################################################################################

# investigating priors

# Prior for sigma: Starting with the exponential distribution
# https://en.wikipedia.org/wiki/Exponential_distribution

?rexp # show help for exponential distribution

# need: value specified for hyperparameter lambda

lambda <- c(.1, .5, 1, 5, 10)
sigma <- seq(0,1000, .01)
y <- cbind(
  dexp(x = sigma, rate = lambda[1]),
  dexp(x = sigma, rate = lambda[2]),
  dexp(x = sigma, rate = lambda[3]),
  dexp(x = sigma, rate = lambda[4]),
  dexp(x = sigma, rate = lambda[5])
)

x <- cbind(sigma, sigma, sigma, sigma, sigma)

# Plot columns of matrix
?matplot

matplot(x = x, y = y, type = "l", lty = 1:5, col = 1:5, lwd = 2)
matplot(x = x, y = y, type = "l", lty = 1:5, col = 1:5, xlim = c(0, 100), 
lwd = 2)
matplot(x = x, y = y, type = "l", lty = 1:5, col = 1:5, xlim = c(0, 10), 
lwd = 2)
matplot(x = x, y = y, type = "l", lty = 1:5, col = 1:5, xlim = c(0, 4), 
lwd = 2)
legend(x = 50, y = 5, legend = paste0(lambda), lty = 1:5, col = 1:5, lwd = 2)


# adding prior variance to sigma
fml_mdl03 <- "

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
  sigma ~ exponential(.1); // prior for sigma
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, sigma);
}

"

# compile model -- this method is for stan code as a string
mdl03 <- cmdstan_model(stan_file = write_stan_file(fml_mdl03))

# run MCMC chain (sample from posterior p.d.)
fit03 = mdl03$sample(
  data = stanls_mdl01,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

###############
# diagnositcs #
###############

fit00$cmdstan_diagnose()
fit00$print()
fit00$diagnostic_summary()

# maximum R-hat
max(fit03$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(fit03$draws())

# posterior histograms
mcmc_hist(fit03$draws())

# posterior densities
mcmc_dens(fit03$draws())

# comparison of results
stanSummary03 = fit03$summary()

# comparison of fixed effects
cbind(stanSummary01[2:8,2:4], stanSummary03[2:8,2:4])

# adding prior to betas
fml_mdl04 = "

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
  betaHeight ~ normal(0,1);
  betaGroup2 ~ normal(0,1);
  betaGroup3 ~ normal(0,1);
  betaHxG2 ~ normal(0,1);
  betaHxG3 ~ normal(0,1);
  
  sigma ~ exponential(.1); // prior for sigma
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, sigma);
}

"

# compile model -- this method is for stan code as a string
mdl04 <- cmdstan_model(stan_file = write_stan_file(fml_mdl04))

# run MCMC chain (sample from posterior p.d.)
fit04 <- mdl04$sample(
  data = stanls_mdl01, 
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

# assess convergence: summary of all parameters
fit04$summary()

# maximum R-hat
max(fit04$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(fit04$draws())

# posterior histograms
mcmc_hist(fit04$draws())

# posterior densities
mcmc_dens(fit04$draws())

# comparison of results
stanSummary04 = fit04$summary()

# comparison of fixed effects
cbind(stanSummary01[2:8,2:4], stanSummary04[2:8,2:4])

################################################################################
# what do you notice being different?
#
# -> Standard error increased dramatically because the prior regulaized the
# estimates to our regularization constraint. Since its too strict, the linear
# predictions become worse – that is the difference between the observed values
# and the linearly predicted values increases. That is the residual standard
# deviation increases.
################################################################################

##############################################
# MATRIX ALGEBRA                             #
# making it easier to supply input into Stan #
##############################################

# adding prior to betas
fml_mdl05a <- "

data {
  int<lower=0> N;         // number of observations
  int<lower=0> P;         // number of predictors (plus column for intercept)
  matrix[N, P] X;         // model.matrix() from R 
  vector[N] y;            // outcome
  
  vector[P] meanBeta;       // prior mean vector for coefficients
  matrix[P, P] covBeta; // prior covariance matrix for coefficients
  
  real sigmaRate;         // prior rate parameter for residual standard deviation
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

# TODO: change! make y = X*beta + alpha + e

fml_mdl05 <- "

data {
  int<lower=0> N;         // number of observations
  int<lower=0> P;         // number of predictors 
  matrix[N, P] X;         // model.matrix() from R 
  vector[N] y;            // outcome
  
  vector[P] mu_beta;       // prior mean vector for coefficients
  matrix[P, P] Sigma_beta; // prior covariance matrix for coefficients
  
  real lambda_sigma;       // prior rate parameter for residual sd 
}

parameters {
  vector[P] beta;         // vector of coefficients for beta
  real<lower=0> sigma;    // residual standard deviation
}

model {
  beta ~ multi_normal(mu_beta, Sigma_beta); // prior for coefficients
  sigma ~ exponential(lambda_sigma);         // prior for sigma
  y ~ normal(X*beta, sigma);        // linear model
}

"

# start with model formula
fml <- formula(WeightLB ~ Height60IN + factor(DietGroup) + Height60IN:factor(DietGroup), data = DietData)

# grab model matrix
(model05_predictorMatrix = model.matrix(fml, data = DietData))
dim(model05_predictorMatrix)

# find details of model matrix
N <- nrow(model05_predictorMatrix)
P <- ncol(model05_predictorMatrix)

# build matrices of hyper parameters (for priors)
mu_beta <- rep(0, P)
Sigma_beta <- diag(x = 10000, nrow = P, ncol = P)
lambda_sigma <- .1

# build Stan data from model matrix
stanls_mdl05 <- list(
  N = N,
  P = P,
  X = model05_predictorMatrix,
  y = DietData$WeightLB,
  mu_beta = mu_beta,
  Sigma_beta = Sigma_beta,
  lambda_sigma = lambda_sigma 
)

mdl05 <- cmdstan_model(stan_file = write_stan_file(fml_mdl05), 
pedantic = TRUE)

fit05 = mdl05$sample(
  data = stanls_mdl05,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000
)

# assess convergence: summary of all parameters
fit05$summary()

# maximum R-hat
max(fit05$summary()$rhat)

# visualize posterior timeseries
mcmc_trace(fit05$draws())

# posterior histograms
mcmc_hist(fit05$draws())

# posterior densities
mcmc_dens(fit05$draws())

# comparison of results
stanSummary05 <- fit05$summary()

# comparison of fixed effects
cbind(stanSummary01[2:8, 2:4], stanSummary05[2:8, 2:4])

# which models fit the best?
# how can we tell?

# next week:
#   model comparison with WAIC
#   posterior predictive model checks
#   combinations of parameters