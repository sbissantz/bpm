
#########
# setup #
#########

library(cmdstanr)
library(bayesplot)
library(ggplot2)
library(posterior)

# set number of cores to 4 for this analysis
options(mc.cores = 4)

########
# data #
########

# model the belief in conspiracy theories
# assuming a normal binomial p.d. for the data
conspiracy_data <- read.csv("./data/conspiracies.csv")

# only using the first 10 items
# positive values mean resemble agreement
conspiracy_items <- conspiracy_data[, 1 : 10]

# Number of items 
I <- ncol(conspiracy_items)

# NEVER DO THIS IN PRACTICE!
# converting to dichotomous responses
# here 0 == strongly disagree or disagree;
# 1 == neither, agree, and strongly disagree
items_bin <- conspiracy_items
for (var in seq_len(10)) {
  items_bin[which(items_bin[, var] <= 3), var] = 0
  items_bin[which(items_bin[, var] > 3), var] = 1
}

# examining data after transformation
table(items_bin$PolConsp1, items_bin$PolConsp1)

# item means
apply(X = items_bin, MARGIN = 2, FUN = mean)

#################################
# example: Likelihood Functions #
#################################

# number of respondents
P <- nrow(items_bin)

# examine the data likelihood for the factor loading of the 1st item, lambda_1

# We have the jont p.m.f. f(Y_p| lambda_1, mu_1 theta_p)
# We fix mu_1 and theta_p
# ...for lambda1
mu1 <- -2 # fix
theta <- rnorm(P, 0, 1) # fix

# Assuming that observations are independent
# f(Y_p| lambda_1, theta_p) = prod_{p=1}^{P} f(Y_p| lambda_1, theta_p)

# We let the loadings vary, to find the maximum (likelihood estimate)
lambda <- seq(-2, 2, .01) # Loadings
log_lik <- vector("numeric", I)

# par <- 1 # for demonstrating
for (par in seq_along(lambda)) {
  # calculate the log-odd or logits
  logit <- mu1 + lambda[par] * theta
  # Convert to probability
  p <- exp(logit) / (1 + exp(logit))
  # Plug the probability into the binomial p.m.f.
  # The product becomes a sum because of the log: log-likelihood
  LL_bern <- sum(dbinom(items_bin$PolConsp1, 1, p, log = TRUE))
  log_lik[par] <- LL_bern
}

# visualize
plot(x = lambda, y = log_lik, type = "l")

# examine the data likelihood for latent trait of the 2nd person, lambda_1

# We have the jont p.m.f. f(Y_p| lambda_1, mu_1 theta_p)
# We fix mu and lambda 
# .... for theta2
mu <- runif(I, -2, 0) # fix
lambda <- runif(I, 0, 2) # fix
person <- 2 # fix

theta <- seq(-3, 3, .01)
log_lik <- NULL
LL_theta <- vector("numeric", P)

par <- 1 # for demonstrating
for (par in seq_along(theta)) {
  for (i in seq_along(I)){
    logit <- mu[i] + lambda[i] * theta[par]
    p <- exp(logit) / (1 + exp(logit))
    LL_theta <- dbinom(items_bin[person, i], 1, p, log = TRUE)
    log_lik[par] <- LL_theta
  }
}

# Visualize
plot(x = theta, y = log_lik, type = "l")

# IRT Model Syntax (slope/intercept form )

fml_2PL_SI = "

data {
  int<lower=0> P;   // number of observations
  int<lower=0> I;   // number of items
  
  // Important: The data are P x I, but we need I x P (row major order)
  array[I, P] int<lower=0, upper=1>  Y; // item responses in an array

  vector[I] mu_mean;             // prior mean vector for intercept parameters
  matrix[I, I] Mu_cov;      // prior covariance matrix for intercept parameters
  
  vector[I] lambda_mean;    // prior mean vector for discrimination parameters
  matrix[I, I] Lambda_cov;  // prior covariance matrix for discrimination parameters
}

parameters {
  vector[P] theta;         // latent variables (one for each person)
  vector[I] mu;            //  item intercepts (one for each item)
  vector[I] lambda;        // factor loadings (one for each item)
}

model { 
  // Prior for item discrimination/factor loadings
  lambda ~ multi_normal(lambda_mean, Lambda_cov); 

  mu ~ multi_normal(mu_mean, Mu_cov); // Prior for item intercepts
  theta ~ normal(0, 1);               // Prior for LV (with mean/sd specified)
  for (i in 1:I){
    
    // Import: If we loop with '[i]' we access every person! (row major order)
    // So the statement is still vectorized because we do not loop over people
    Y[i] ~ bernoulli_logit(mu[i] + lambda[i]*theta);
  }
}

"

# compile model
mdl_2PL_SI <- cmdstan_model(stan_file = write_stan_file(fml_2PL_SI))

# data dimensions
P <- nrow(conspiracy_items)

# item intercept hyperparameters
mu_mean_hp <- 0
mu_mean <- rep(mu_mean_hp, I)

mu_var_hp <- 1000
Mu_cov <- diag(mu_var_hp, I)

# item discrimination/factor loading hyperparameters
lambda_mean_hp <- 0
lambda_mean <- rep(lambda_mean_hp, I)

lambda_var_hp <- 1000
Lambda_cov <- diag(lambda_var_hp, I)

#############
# stan list #
#############

# build r list for stan
stanls_2PL_SI <- list(
  "P" = P,
  "I" = I,
  # Important transpose (array in stan are in row major order)
  "Y" = t(items_bin),
  "mu_mean" = mu_mean,
  "Mu_cov" = Mu_cov,
  "lambda_mean" = lambda_mean,
  "Lambda_cov" = Lambda_cov 
)

# run MCMC chain (sample from posterior p.d.)
fit_2PL_SI <- mdl_2PL_SI$sample(
  data = stanls_2PL_SI,
  seed = 02112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 3000,
  # Mean should be below 10, since the log of it is too large
  init = function() list(lambda = rnorm(I, mean = 5, sd = 1))
)

# assess convergence: summary of all parameters
fit_2PL_SI$summary()
fit_2PL_SI$cmdstan_diagnose()
fit_2PL$diagnostic_summary()

# checking convergence
max(fit_2PL_SI$summary()$rhat, na.rm = TRUE)

# item parameter results
print(fit_2PL_SI$summary(variables = c("mu", "lambda")), n = Inf)

###################
# item parameters #
###################

# summary of the item parameters
fit_2PL_SI$summary(variables = "mu") # E(Y| theta = 0)
fit_2PL_SI$summary(variables = "lambda") # E(Y| theta + 1) - E(Y| theta)

# extract posterior draws
draws <- posterior::as_draws_rvars(fit_2PL_SI$draws())

# fixed theta values
theta_fixed <- seq(-3, 3, length.out = P)

# drawing item characteristic curves for item
draws$logit <- draws$mu + draws$lambda * t(theta_fixed)
# ...including estimation uncertainty in theta
# draws$logit <- draws$mu + draws$lambda * t(draws$theta)

# Cannot use logistic function directly, because of the rvar data type
draws$y <- exp(draws$logit) / (1 + exp(draws$logit))

# Visualize the item characteristic curve for item 5
itemno <- 5
plot(
  x = theta_fixed, y = mean(draws$y[itemno, ]), type = "l",
  main = paste("Item", itemno, "ICC"), ylim = c(0, 1), lwd = 2,
  xlab = expression(theta),
  ylab = paste("Item", itemno, "Retrodicted Value")
)
yno_arr <- posterior::draws_of(draws$y[itemno, ])
for (d in 1:100) {
  lines(theta_fixed, yno_arr[d, 1, ], col = "steelblue", lwd = 0.5)
}
lines(theta_fixed, mean(draws$y[itemno, ]), lwd = 5)
legend(-3, 1,
  legend = c("Posterior Draw", "EAP"),
  col = c("steelblue", "black"), lty = c(1, 1), lwd = 5
)

# investigating item parameters
#

# item intercepts
mcmc_trace(fit_2PL_SI$draws(variables = "mu"))
mcmc_dens(fit_2PL_SI$draws(variables = "mu"))
# Results are pretty skewed

# loadings
mcmc_trace(fit_2PL_SI$draws(variables = "lambda"))
mcmc_dens(fit_2PL_SI$draws(variables = "lambda"))
# Results are pretty skewed

# bivariate posterior p.d.
mcmc_pairs(fit_2PL_SI$draws(), pars = c("mu[1]", "lambda[1]"))
# Even though we specified the prior p.d.s for the parameter independently, the
# posterior p.d.s are not independent

# investigating the latent variables
fit_2PL_SI$summary(variables = "theta")

# EAP Estimates of Latent Variables
hist(mean(draws$theta),
  main = "EAP Estimates of Theta",
  xlab = expression(theta)
)

# Comparing two posterior distributions
plot(c(-3, 3), c(0, 2), type = "n", xlab = expression(theta), ylab = "Density")
lines(density(draws_of(draws$theta[1])), col = "red", lwd = 3)
lines(density(draws_of(draws$theta[2])), col = "blue", lwd = 3)

# Comparing EAP Estimates with Posterior SDs
plot(y = sd(draws$theta), x = mean(draws$theta), pch = 19,
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)", 
     main = "Mean vs SD of Theta")

# Comparing EAP Estimates with Sum Scores
plot(y = rowSums(items_bin), x = mean(draws$theta), pch = 19,
     ylab = "Sum Score", xlab = expression(theta))

# IRT Model Syntax (slope/intercept form with discrimination/difficulty calculated) ===========

# TODO

fml_2PL_SI2 <- "

data {
  int<lower=0> P;    // number of observations
  int<lower=0> I;  // number of items

  // Important: The data are P x I, but we need I x P (row major order)
  array[I, P] int<lower=0, upper=1>  Y; // item responses in an array

  vector[nItems] meanMu;             // prior mean vector for intercept parameters
  matrix[nItems, nItems] covMu;      // prior covariance matrix for intercept parameters
  
  vector[nItems] meanLambda;         // prior mean vector for discrimination parameters
  matrix[nItems, nItems] covLambda;  // prior covariance matrix for discrimination parameters
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] mu;                 // the item intercepts (one for each item)
  vector[nItems] lambda;             // the factor loadings/item discriminations (one for each item)
}

model {
  
  lambda ~ multi_normal(meanLambda, covLambda); // Prior for item discrimination/factor loadings
  mu ~ multi_normal(meanMu, covMu);             // Prior for item intercepts
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli_logit(mu[item] + lambda[item]*theta);
  }
  
}

generated quantities{
  vector[nItems] a;
  vector[nItems] b;
  
  for (item in 1:nItems){
    a[item] = lambda[item];
    b[item] = -1*mu[item]/lambda[item];
  }
  
}

"

modelIRT_2PL_SI2_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_2PL_SI2_syntax))


modelIRT_2PL_SI2_samples = modelIRT_2PL_SI2_stan$sample(
  data = modelIRT_2PL_SI_data,
  seed = 02112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelIRT_2PL_SI2_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelIRT_2PL_SI2_samples$summary(variables = c("a", "b")) ,n=Inf)

# IRT Model Syntax (discrimination/difficulty form ) ==================================================

modelIRT_2PL_DD_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in a matrix

  vector[nItems] meanA;
  matrix[nItems, nItems] covA;      // prior covariance matrix for coefficients
  
  vector[nItems] meanB;         // prior mean vector for coefficients
  matrix[nItems, nItems] covB;  // prior covariance matrix for coefficients
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] a;                 // the item intercepts (one for each item)
  vector[nItems] b;             // the factor loadings/item discriminations (one for each item)
}

model {
  
  a ~ multi_normal(meanA, covA); // Prior for item discrimination/factor loadings
  b ~ multi_normal(meanB, covB);             // Prior for item intercepts
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli_logit(a[item]*(theta - b[item]));
  }
  
}

generated quantities{
  vector[nItems] lambda;
  vector[nItems] mu;
  
  lambda = a;
  for (item in 1:nItems){
    mu[item] = -1*a[item]*b[item];
  }
}

"

modelIRT_2PL_DD_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_2PL_DD_syntax))

# data dimensions
nObs = nrow(conspiracyItems)
nItems = ncol(conspiracyItems)

# item intercept hyperparameters
bMeanHyperParameter = 0
bMeanVecHP = rep(bMeanHyperParameter, nItems)

bVarianceHyperParameter = 1000
bCovarianceMatrixHP = diag(x = bVarianceHyperParameter, nrow = nItems)

# item discrimination/factor loading hyperparameters
aMeanHyperParameter = 0
aMeanVecHP = rep(aMeanHyperParameter, nItems)

aVarianceHyperParameter = 1000
aCovarianceMatrixHP = diag(x = aVarianceHyperParameter, nrow = nItems)

modelIRT_2PL_DD_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = t(conspiracyItemsDichtomous), 
  meanB = bMeanVecHP,
  covB = bCovarianceMatrixHP,
  meanA = aMeanVecHP,
  covA = aCovarianceMatrixHP
)

modelIRT_2PL_DD_samples = modelIRT_2PL_DD_stan$sample(
  data = modelIRT_2PL_DD_data,
  seed = 02112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(a=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelIRT_2PL_DD_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelIRT_2PL_DD_samples$summary(variables = c("a", "b")) ,n=Inf)

# comparing with other parameters estimated:
plot(x = modelIRT_2PL_DD_samples$summary(variables = c("b"))$mean,
     y = modelIRT_2PL_SI2_samples$summary(variables = c("b"))$mean,
     xlab = "Discrimination/Difficulty Model", 
     ylab = "Slope/Intercept Model",
     main = "Difficulty Parameter EAP Estimates"
)

# comparing with other parameters estimated:
plot(x = modelIRT_2PL_DD_samples$summary(variables = c("a"))$mean,
     y = modelIRT_2PL_SI2_samples$summary(variables = c("a"))$mean,
     xlab = "Discrimination/Difficulty Model", 
     ylab = "Slope/Intercept Model",
     main = "Discrimination Parameters EAP Estimates"
)

# theta results
# comparing with other parameters estimated:
plot(x = modelIRT_2PL_DD_samples$summary(variables = c("theta"))$mean,
     y = modelIRT_2PL_SI2_samples$summary(variables = c("theta"))$mean,
     xlab = "Discrimination/Difficulty Model", 
     ylab = "Slope/Intercept Model",
     main = "Theta EAP Estimates"
)

# comparing with other parameters estimated:
plot(x = modelIRT_2PL_DD_samples$summary(variables = c("theta"))$sd,
     y = modelIRT_2PL_SI2_samples$summary(variables = c("theta"))$sd,
     xlab = "Discrimination/Difficulty Model", 
     ylab = "Slope/Intercept Model",
     main = "Theta SD Estimates"
)


# IRT Auxiliary Statistics ===========================================================

modelIRT_2PL_DD2_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in a matrix

  vector[nItems] meanA;
  matrix[nItems, nItems] covA;      // prior covariance matrix for coefficients
  
  vector[nItems] meanB;         // prior mean vector for coefficients
  matrix[nItems, nItems] covB;  // prior covariance matrix for coefficients
  
  int<lower=0> nThetas;        // number of theta values for auxiliary statistics
  vector[nThetas] thetaVals;   // values for auxiliary statistics
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] a;                 // the item intercepts (one for each item)
  vector[nItems] b;             // the factor loadings/item discriminations (one for each item)
}

model {
  
  a ~ multi_normal(meanA, covA); // Prior for item discrimination/factor loadings
  b ~ multi_normal(meanB, covB);             // Prior for item intercepts
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli_logit(a[item]*(theta - b[item]));
  }
  
}

generated quantities{
  vector[nItems] lambda;
  vector[nItems] mu;
  vector[nThetas] TCC;
  matrix[nThetas, nItems] itemInfo;
  vector[nThetas] testInfo;
  
  for (val in 1:nThetas){
    TCC[val] = 0.0;
    testInfo[val] = -1.0;  // test information must start at -1 to include prior distribution for theta
    for (item in 1:nItems){
      itemInfo[val, item] = 0.0;
    }
  }
  
  lambda = a;
  for (item in 1:nItems){
    mu[item] = -1*a[item]*b[item];
    
    for (val in 1:nThetas){
      // test characteristic curve:
      TCC[val] = TCC[val] + inv_logit(a[item]*(thetaVals[val]-b[item]));
      
      // item information functions:
      itemInfo[val, item] = 
        itemInfo[val, item] + 
          a[item]^2*inv_logit(a[item]*(thetaVals[val]-b[item]))*(1-inv_logit(a[item]*(thetaVals[val]-b[item])));
        
      // test information functions:
      testInfo[val] = testInfo[val] + 
        a[item]^2*inv_logit(a[item]*(thetaVals[val]-b[item]))*(1-inv_logit(a[item]*(thetaVals[val]-b[item])));
    }
  }
  
  
  
}

"

modelIRT_2PL_DD2_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_2PL_DD2_syntax))

thetaVals = seq(-3,3,.01)

modelIRT_2PL_DD2_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = t(conspiracyItemsDichtomous), 
  meanB = bMeanVecHP,
  covB = bCovarianceMatrixHP,
  meanA = aMeanVecHP,
  covA = aCovarianceMatrixHP,
  nThetas = length(thetaVals),
  thetaVals = thetaVals
)

modelIRT_2PL_DD2_samples = modelIRT_2PL_DD2_stan$sample(
  data = modelIRT_2PL_DD2_data,
  seed = 02112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(a=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelIRT_2PL_DD2_samples$summary(variables = c("theta", "a", "b"))$rhat, na.rm = TRUE)

# item parameter results
print(modelIRT_2PL_DD2_samples$summary(variables = c("TCC")) ,n=Inf)


# TCC Spaghetti Plots
tccSamples = modelIRT_2PL_DD2_samples$draws(variables = "TCC", format = "draws_matrix")
plot(x = thetaVals, 
     y = tccSamples[1,],
     xlab = expression(theta), 
     ylab = "Expected Score", type = "l",
     main = "Test Characteristic Curve", lwd = 2)

for (draw in 1:nrow(tccSamples)){
  lines(x = thetaVals,
        y = tccSamples[draw,])
}

# EAP TCC
lines(x = thetaVals, 
      y = modelIRT_2PL_DD2_samples$summary(variables = c("TCC"))$mean,
      lwd = 2, 
      col=2, 
      lty=3)

legend(x = -3, y = 7, legend = c("Posterior Draw", "EAP"), col = c(1,2), lty = c(1,2), lwd=5)

# ICC Spaghetti Plots
item = 1
itemLabel = paste0("Item ", item)
iccSamples = modelIRT_2PL_DD2_samples$draws(variables = "itemInfo", format = "draws_matrix")
iccNames = colnames(iccSamples)
itemSamples = iccSamples[,iccNames[grep(pattern = ",1]", x = iccNames)]]

maxInfo = max(apply(X = itemSamples, MARGIN = 2, FUN = max))

plot(x = thetaVals, 
     y = itemSamples[1,],
     xlab = expression(theta), 
     ylab = "Information", type = "l",
     main = paste0(itemLabel, " Information Function"), lwd = 2,
     ylim = c(0,maxInfo+.5))

for (draw in 1:nrow(itemSamples)){
  lines(x = thetaVals,
        y = itemSamples[draw,])
}

# EAP TCC
lines(x = thetaVals, 
      y = apply(X = itemSamples, MARGIN=2, FUN=mean),
      lwd = 3, 
      col = 2, 
      lty = 3)

legend(x = -3, y = maxInfo-.5, legend = c("Posterior Draw", "EAP"), col = c(1,2), lty = c(1,2), lwd=5)


# TIF Spaghetti Plots
tifSamples = modelIRT_2PL_DD2_samples$draws(variables = "testInfo", format = "draws_matrix")
maxTIF = max(apply(X = tifSamples, MARGIN = 2, FUN = max))

plot(x = thetaVals, 
     y = tifSamples[1,],
     xlab = expression(theta), 
     ylab = "Information", type = "l",
     main = "Test Information Function", lwd = 2,
     ylim = c(0,maxTIF))

for (draw in 1:nrow(tifSamples)){
  lines(x = thetaVals,
        y = tifSamples[draw,])
}

# EAP TIF
lines(x = thetaVals, 
      y = apply(X=tifSamples, MARGIN=2, FUN=mean),
      lwd = 3, 
      col = 2, 
      lty = 3)

legend(x = -3, y = maxTIF, legend = c("Posterior Draw", "EAP"), col = c(1,2), lty = c(1,2), lwd=5)

# EAP TCC
plot(x = thetaVals, 
     y = apply(X=tifSamples, MARGIN=2, FUN=mean),
     xlab = expression(theta), 
     ylab = "Information", type = "l",
     main = "Test Information Function", 
     lwd = 2)

# Other IRT Models ===========================================================

# 1PL Model:
modelIRT_1PL_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in a matrix
  
  vector[nItems] meanB;         // prior mean vector for coefficients
  matrix[nItems, nItems] covB;  // prior covariance matrix for coefficients
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] b;             // the factor loadings/item discriminations (one for each item)
}

model {
  b ~ multi_normal(meanB, covB);             // Prior for item intercepts
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli(inv_logit(theta - b[item]));
  }
  
}

"

modelIRT_1PL_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_1PL_syntax))

modelIRT_1PL_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = t(conspiracyItemsDichtomous), 
  meanB = bMeanVecHP,
  covB = bCovarianceMatrixHP
)

modelIRT_1PL_samples = modelIRT_1PL_stan$sample(
  data = modelIRT_1PL_data,
  seed = 021120221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000
)

# checking convergence
max(modelIRT_1PL_samples$summary(variables = c("theta", "b"))$rhat, na.rm = TRUE)

modelIRT_1PL_samples$summary(variables = c("b"))

# 3PL Model:
modelIRT_3PL_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in a matrix
  
  vector[nItems] meanA;
  matrix[nItems, nItems] covA;      // prior covariance matrix for coefficients
  
  vector[nItems] meanB;         // prior mean vector for coefficients
  matrix[nItems, nItems] covB;  // prior covariance matrix for coefficients
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] a;
  vector[nItems] b;             // the factor loadings/item discriminations (one for each item)
  vector<lower=0, upper=1>[nItems] c;
}

model {
  a ~ multi_normal(meanA, covA);             // Prior for item intercepts
  b ~ multi_normal(meanB, covB);             // Prior for item intercepts
  c ~ beta(1,1);                              // Simple prior for c parameter
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli(c[item] + (1-c[item])*inv_logit(a[item]*(theta - b[item])));
  }
  
}

"

modelIRT_3PL_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_3PL_syntax))

modelIRT_3PL_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = t(conspiracyItemsDichtomous), 
  meanB = bMeanVecHP,
  covB = bCovarianceMatrixHP,
  meanA = aMeanVecHP,
  covA = aCovarianceMatrixHP
)

modelIRT_3PL_samples = modelIRT_3PL_stan$sample(
  data = modelIRT_3PL_data,
  seed = 021120222,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(a=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelIRT_3PL_samples$summary(variables = c("theta", "b", "a", "c"))$rhat, na.rm = TRUE)

print(modelIRT_3PL_samples$summary(variables = c("a", "b", "c")), n=Inf)

# Two-Parameter Normal Ogive Model:
modelIRT_2PNO_syntax = "

data {
  int<lower=0> nObs;                 // number of observations
  int<lower=0> nItems;               // number of items
  array[nItems, nObs] int<lower=0, upper=1>  Y; // item responses in a matrix
  
  vector[nItems] meanA;
  matrix[nItems, nItems] covA;      // prior covariance matrix for coefficients
  
  vector[nItems] meanB;         // prior mean vector for coefficients
  matrix[nItems, nItems] covB;  // prior covariance matrix for coefficients
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  vector[nItems] a;
  vector[nItems] b;             // the factor loadings/item discriminations (one for each item)
}

model {
  a ~ multi_normal(meanA, covA);             // Prior for item intercepts
  b ~ multi_normal(meanB, covB);             // Prior for item intercepts
  
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    Y[item] ~ bernoulli(Phi(a[item]*(theta - b[item])));
  }
  
}

"

modelIRT_2PNO_stan = cmdstan_model(stan_file = write_stan_file(modelIRT_2PNO_syntax))

modelIRT_2PNO_data = list(
  nObs = nObs,
  nItems = nItems,
  Y = t(conspiracyItemsDichtomous), 
  meanB = bMeanVecHP,
  covB = diag(nItems),
  meanA = aMeanVecHP,
  covA = diag(nItems) # changing prior covariance to help with convergence
)

modelIRT_2PNO_samples = modelIRT_2PNO_stan$sample(
  data = modelIRT_2PNO_data,
  seed = 0211202223,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(a=rnorm(nItems, mean=3, sd=.05))
)

# checking convergence -- not great!
max(modelIRT_2PNO_samples$summary(variables = c("theta", "b", "a"))$rhat, na.rm = TRUE)

print(modelIRT_2PNO_samples$summary(variables = c("a", "b")), n=Inf)


save.image("lecture04c.RData")

