# ###################################
# Bayesian Model Fit and Comparison #
# ###################################

library(cmdstanr)
library(bayesplot)
library(ggplot2)
library(loo)

########
# data #
########

# outcome: Weight in pounds
DietData <- read.csv(file = "./data/DietData.csv")

# important that we know what 0 is in our interaction
# center predictor variable
# gives E(Y | X = 60) NOT 0 anymore
DietData$Height60IN <- DietData$HeightIN - 60

# dummy variable for group 2
group2 <- rep(0, nrow(DietData))
group2[which(DietData$DietGroup == 2)] <- 1

# dummy variable for group 3
group3 <- rep(0, nrow(DietData))
group3[which(DietData$DietGroup == 3)] <- 1

# interaction terms
heightXgroup2 <- DietData$Height60IN * group2
heightXgroup3 <- DietData$Height60IN * group3

# our matrix syntax from before, but now with PPMC built in

fml06 <- "

data {
  int<lower=0> N;         // number of observations
  int<lower=0> P;         // number of predictors (plus column for intercept)
  matrix[N, P] X;         // model.matrix() from R 
  vector[N] y;            // outcome
  
  vector[P] mu_beta;       // prior mean vector for coefficients
  matrix[P, P] Sigma_beta; // prior covariance matrix for coefficients
  
  real lambda_sigma;       // prior rate parameter for residual sd 
}

parameters {
  vector[P] beta;         // vector of coefficients for Beta
  real<lower=0> sigma;    // residual standard deviation
}

model {
  beta ~ multi_normal(mu_beta, Sigma_beta); // prior for coefficients
  sigma ~ exponential(lambda_sigma);         // prior for sigma
  y ~ normal(X*beta, sigma);              // linear model
}

generated quantities{

  // general quantities used below:
  vector[N] y_pred;
  y_pred = X*beta; // predicted value (conditional mean)

  // posterior predictive model checking
  
  array[N] real y_sim; //stan wants an array here
  y_sim = normal_rng(y_pred, sigma);
  
  real mean_y = mean(y);
  real sd_y = sd(y);
  real mean_y_sim = mean(to_vector(y_sim));
  real<lower=0> sd_y_sim = sd(to_vector(y_sim));

  //posterior predictive p values
  int<lower=0, upper=1> mean_gte = (mean_y_sim >= mean_y);
  int<lower=0, upper=1> sd_gte = (sd_y_sim >= sd(y));
  
  // WAIC and LOO for model comparison
 
  array[N] real log_lik; //stan wants an array here
  // Import: We calculate the log likelihood for each person  
  for (person in 1:N){
    log_lik[person] = normal_lpdf(y[person] | y_pred[person], sigma);
  }
}

"

# compile stan code into executable
mdl06 <- cmdstan_model(stan_file = write_stan_file(fml06))

# start with model formula
fml <- formula(WeightLB ~ Height60IN + factor(DietGroup) + 
Height60IN:factor(DietGroup), data = DietData)

temp <- lm(fml, data = DietData)
plot(temp)
# grab model matrix
X06 <- model.matrix(fml, data = DietData)

# find details of model matrix
(N <- nrow(X06))
(P <- ncol(X06))

# build matrices of hyper parameters (for priors)
(mu_beta <- rep(0, P))
(Sigma_beta <- diag(x = 1000, nrow = P, ncol = P))
(lambda_sigma <- .1)

# build Stan data from model matrix
stanls06 <- list(
  N = N,
  P = P,
  X = X06,
  y = DietData$WeightLB,
  mu_beta = mu_beta,
  Sigma_beta = Sigma_beta,
  lambda_sigma = lambda_sigma 
)

fit06 <- mdl06$sample(
  data = stanls06,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000
)
?draws
# here, we use format = "draws_matrix" to remove the draws from the array
# format they default to
postdraws <- fit06$draws(c("beta", "sigma"), format = "draws_matrix")
View(postdraws)

# Posterior predictive checks in R
# Here, we build a predictive distribution

# 1. Take a random draw from posterior (we do this actually for all draws)
(iter <- sample(x = seq_len(nrow(postdraws)), size = 1, replace = TRUE))

# 2. Get the one draw from posterior for all parameters
postdraws[iter, ]

# vector of estimated coefficients for this draw
(beta_hat <- matrix(data = postdraws[iter, 1:6], ncol = 1))

# estimated residual standard deviation 
(sigma_hat <- postdraws[iter, 7])

# conditional means E(Y | X) for this draw
# y_hat = X %*% beta_hat
(condmeans <- X06 %*% beta_hat) 

# simulate data from posterior predictive distribution for this draw
simdat <- rnorm(n = N, mean = condmeans, sd = sigma_hat)
hist(simdat)

(simean <- mean(simdat)) ; mean(DietData$WeightLB)
(simsd <- sd(simdat)) ; sd(DietData$WeightLB)

# Now, we do this for all draws (and all data points)
#
simean <- rep(NA, nrow(postdraws))
simsd <- rep(NA, nrow(postdraws))
for (i in seq_len(nrow(postdraws))){
  beta_hat <- matrix(data = postdraws[i, 1:6], ncol = 1)
  sigma_hat <- postdraws[i, 7]
  condmeans <- model06_predictorMatrix %*% beta_hat
  simdat <- rnorm(n = N, mean = condmeans, sd = sigma_hat)
  simean[i] <- mean(simdat)
  simsd[i] <- sd(simdat)
}

# Posterior predictive mean checking
hist(simean)

# maximum R-hat
max(fit06$summary()$rhat, na.rm = TRUE)

# show results
View(fit06$summary())

# posterior predictive histograms for data points 1 and 30
mcmc_hist(fit06$draws("y_sim[1]")) + 
  geom_vline(xintercept = DietData$WeightLB[1], color = "orange")
mcmc_hist(fit06$draws("y_sim[30]")) + 
  geom_vline(xintercept = DietData$WeightLB[30], color = "orange")

# posterior predictive histograms for statistics of y
mcmc_hist(fit06$draws("mean_y_sim")) + 
  geom_vline(xintercept = mean(DietData$WeightLB), color = "orange")
mcmc_hist(fit06$draws("sd_y_sim")) + 
  geom_vline(xintercept = sd(DietData$WeightLB), color = "orange")

# calculate WAIC for model comparisons
waic(x = fit06$draws("log_lik"))

# calculate LOO for model comparisons
fit06$loo()

# next: try more informative prior p.d.
# build matrices of hyper parameters (for priors)
mu_beta <- rep(0, P)
Sigma_beta <- diag(x = 1, nrow = P, ncol = P)
lambda_sigma <- 10

# build Stan data from model matrix
stanls06b <- list(
  N = N,
  P = P,
  X = model06_predictorMatrix,
  y = DietData$WeightLB,
  mu_beta = mu_beta,
  Sigma_beta = Sigma_beta,
  lambda_sigma = lambda_sigma 
)

# Important: No need to recompile! Awesome.

fit06b <- mdl06$sample(
  data = stanls06b,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000
)

# maximum R-hat
max(fit06b$summary()$rhat, na.rm=TRUE)

# show results
View(fit06b$summary())

# calculate WAIC for model comparisons
waic(x = fit06$draws("log_lik"))
waic(x = fit06b$draws("log_lik"))

# calculate LOO for model comparisons
fit06b$loo()

# comparing two models with loo:
loo_compare(list(
  "uniformative prior" = fit06$loo(), 
  "informative prior" = fit06b$loo())
)

# final comparison: investigating homogeneity of variance assumption

fml07 <- "

data {
  int<lower=0> N;         // number of observations
  int<lower=0> P;         // number of predictors (plus column for intercept)
  matrix[N, P] X;         // model.matrix() from R 
  vector[N] y;            // outcome
  
  vector[P] mu_beta;       // prior mean vector for coefficients
  matrix[P, P] Sigma_beta; // prior covariance matrix for coefficients
  
  vector[P] mu_gamma;       // prior mean vector for coefficients
  matrix[P, P] Sigma_gamma; // prior covariance matrix for coefficients
}


parameters {
  vector[P] beta;         // vector of coefficients for conditional mean
  vector[P] gamma;        // vector of coefficients for residual sd
}


model {
  beta ~ multi_normal(mu_beta, Sigma_beta); // prior for coefficients
  gamma ~ multi_normal(mu_gamma, Sigma_gamma); // prior for coefficients
  y ~ normal( X*beta, exp(X*gamma) );              // linear model
}

generated quantities{

  // general quantities used below:
  vector[N] y_pred;
  vector[N] y_sd;
  y_pred = X*beta; // predicted value (conditional mean)
  y_sd = exp(X*gamma); // conditional standard deviation
  
  // posterior predictive model checking
  
  array[N] real y_sim;
  y_sim = normal_rng(y_pred, y_sd);
  
  real mean_y = mean(y);
  real sd_y = sd(y);
  real mean_y_sim = mean(to_vector(y_sim));
  real<lower=0> sd_y_sim = sd(to_vector(y_sim));
  int<lower=0, upper=1> mean_gte = (mean_y_sim >= mean_y);
  int<lower=0, upper=1> sd_gte = (sd_y_sim >= sd(y));
  
  // WAIC and LOO for model comparison
  
  array[N] real log_lik;
  for (person in 1:N){
    log_lik[person] = normal_lpdf(y[person] | y_pred[person], y_sd[person]);
  }
}

"

# compile stan code into executable
mdl07 <- cmdstan_model(stan_file = write_stan_file(fml07))

# start with model formula
fml <- formula(WeightLB ~ Height60IN + factor(DietGroup) + Height60IN:factor(DietGroup), data = DietData)

# grab model matrix
X07 = model.matrix(fml, data=DietData)

# find details of model matrix
N = nrow(X07)
P = ncol(X07)

# build matrices of hyper parameters (for priors)
(mu_beta = rep(0, P))
(Sigma_beta = diag(x = 1000, nrow = P, ncol = P))
(mu_gamma <- rep(0, P))
(Sigma_gamma = diag(x = 1000, nrow = P, ncol = P))

# build Stan data from model matrix
stanls07 = list(
  "N" = N,
  "P" = P,
  "X" = X07,
  "y" = DietData$WeightLB,
  "mu_beta" = mu_beta,
  "Sigma_beta" = Sigma_beta,
  "mu_gamma" = mu_gamma,
  "Sigma_gamma" = Sigma_gamma
)

fit07 = mdl07$sample(
  data = stanls07,
  seed =  112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 4000
)

# maximum R-hat
max(fit07$summary()$rhat, na.rm = TRUE)

# model results
View(fit07$summary())
fit07$summary(variables = "gamma")

# model comparisons
waic(x = fit06$draws("log_lik"))
waic(x = fit07$draws("log_lik"))

fit07$loo()
fit07$loo()

loo_compare(list(
  "homogeneous" = fit06$loo(), 
  "heterogeneous" = fit07$loo())
)