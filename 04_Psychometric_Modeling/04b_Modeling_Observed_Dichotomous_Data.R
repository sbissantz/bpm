
#########
# setup #
#########

library(cmdstanr)
library(bayesplot)
library(ggplot2)
library(posterior)

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
# number of respondents
P <- nrow(items_bin)

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
colMeans(items_bin)

#################################
# Example: Likelihood Functions #
#################################

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

################
# IRT: 2 PL SI #
################

# compile model
mdl_2pl_si <- cmdstan_model("./stan/4a/2pl_si.stan", pedantic = TRUE)

# item intercept hyperparameters
mu_mean <- rep(0, I)
Mu_cov <- diag(1000, I)

# item discrimination/factor loading hyperparameters
lambda_mean <- rep(0, I)
Lambda_cov <- diag(1000, I)

#############
# stan list #
#############

# build r list for stan
stanls_2pl_si <- list(
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
fit_2pl_si <- mdl_2pl_si$sample(
  data = stanls_2pl_si,
  seed = 02112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  # Mean should be below 10, since the log of it is too large
  init = function() list(lambda = rnorm(I, mean = 5, sd = 1))
)

# assess convergence: summary of all parameters
fit_2pl_si$summary()
fit_2pl_si$cmdstan_diagnose()
fit_2pl_si$diagnostic_summary()

# checking convergence
max(fit_2pl_si$summary()$rhat, na.rm = TRUE)

# item parameter results
print(fit_2pl_si$summary(variables = c("mu", "lambda")), n = Inf)

###################
# item parameters #
###################

# summary of the item parameters
fit_2pl_si$summary(variables = "mu") # E(Y| theta = 0)
fit_2pl_si$summary(variables = "lambda") # E(Y| theta + 1) - E(Y| theta)

# extract posterior draws
drawsSI <- posterior::as_draws_rvars(fit_2pl_si$draws())

# fixed theta values
theta_fixed <- seq(-3, 3, length.out = P)

# drawing item characteristic curves for item
drawsSI$logit <- drawsSI$mu + drawsSI$lambda * t(theta_fixed)
# ...including estimation uncertainty in theta
# draws$logit <- draws$mu + draws$lambda * t(draws$theta)

# Cannot use logistic function directly, because of the rvar data type
drawsSI$y <- exp(drawsSI$logit) / (1 + exp(drawsSI$logit))

# Visualize the item characteristic curve for item 5
itemno <- 5
plot(
  x = theta_fixed, y = mean(drawsSI$y[itemno, ]), type = "l",
  main = paste("Item", itemno, "ICC"), ylim = c(0, 1), lwd = 2,
  xlab = expression(theta),
  ylab = paste("Item", itemno, "Retrodicted Value")
)
yno_arr <- posterior::draws_of(drawsSI$y[itemno, ])
for (d in 1:100) {
  lines(theta_fixed, yno_arr[d, 1, ], col = "steelblue", lwd = 0.5)
}
lines(theta_fixed, mean(drawsSI$y[itemno, ]), lwd = 5)
legend(-3, 1,
  legend = c("Posterior Draw", "EAP"),
  col = c("steelblue", "black"), lty = c(1, 1), lwd = 5
)

# investigating item parameters
#

# item intercepts
mcmc_trace(fit_2pl_si$draws(variables = "mu"))
mcmc_dens(fit_2pl_si$draws(variables = "mu"))
# Results are pretty skewed

# loadings
mcmc_trace(fit_2pl_si$draws(variables = "lambda"))
mcmc_dens(fit_2pl_si$draws(variables = "lambda"))
# Results are pretty skewed

# bivariate posterior p.d.
mcmc_pairs(fit_2pl_si$draws(), pars = c("mu[1]", "lambda[1]"))
# Even though we specified the prior p.d.s for the parameter independently, the
# posterior p.d.s are not independent

# investigating the latent variables
fit_2pl_si$summary(variables = "theta")

# EAP Estimates of Latent Variables
hist(mean(draws_si$theta),
  main = "EAP Estimates of Theta",
  xlab = expression(theta)
)

# Comparing two posterior distributions
plot(c(-3, 3), c(0, 2), type = "n", xlab = expression(theta), ylab = "Density")
lines(density(draws_of(draws_si$theta[1])), col = "red", lwd = 3)
lines(density(draws_of(draws_si$theta[2])), col = "blue", lwd = 3)

# Comparing EAP Estimates with Posterior SDs
plot(y = sd(draws_si$theta), x = mean(draws_si$theta), pch = 19,
  xlab = "E(theta|Y)", ylab = "SD(theta|Y)", 
  main = "Mean vs SD of Theta")

# Comparing EAP Estimates with Sum Scores
plot(y = rowSums(items_bin), x = mean(draws_si$theta), pch = 19,
  ylab = "Sum Score", xlab = expression(theta))

##################
# IRT: 2PL SI II #
##################
# Discrimination/difficulty calculated in the generated quantities block

# compile model
mdl_2pl_si2 <- cmdstan_model("./stan/4a/2pl_si2.stan", pedantic = TRUE)

# fit model to data
fit_2pl_si2 <- mdl_2pl_si2$sample(
  data = stanls_2pl_si,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  init = function() list(lambda = rnorm(I, mean = 5, sd = 1))
)

# diagnostics 
max(fit_2pl_si2$summary()$rhat, na.rm = TRUE)
fit_2pl_si2$cmdstan_diagnose()
fit_2pl_si2$diagnostic_summary()

# item parameter results
print(fit_2pl_si2$summary(variables = c("a", "b")), n = Inf)

# extract posterior draws
draws_si2 <- posterior::as_draws_rvars(fit_2pl_si2$draws())

#  2PL discrimination/difficulty

fml_2pl_dd <- "

data {
  int<lower=0> P;                 // number of observations
  int<lower=0> I;               // number of items
  array[I, P] int<lower=0, upper=1>  Y; // item responses in a matrix

  vector[I] a_mean;
  matrix[I, I] A_cov;      // prior covariance matrix for coefficients
  
  vector[I] b_mean;         // prior mean vector for coefficients
  matrix[I, I] B_cov;  // prior covariance matrix for coefficients
}

parameters {
  vector[P] theta;                // LV (1/person)
  vector[I] a;                 // item intercepts (1/item)
  vector[I] b;             // item discriminations/factor loading (1/item)
}

model {
  // Prior for item discrimination/factor loadings
  a ~ multi_normal(a_mean, A_cov); 
  b ~ multi_normal(b_mean, B_cov);             // Prior for item intercepts
  
  theta ~ normal(0, 1);    // Standadardized LV (with mean/sd specified)
  
  for (i in 1:I){
    Y[i] ~ bernoulli_logit(a[i]*(theta - b[i]));
  }
  
}

generated quantities{
  vector[I] lambda;
  vector[I] mu;
  
  lambda = a;
  for (i in 1:I){
    mu[i] = -1*a[i]*b[i];
  }
}

"
# compile modeo
mdl_2pl_dd <- cmdstan_model(stan_file = write_stan_file(fml_2pl_dd))

# item intercept hyperparameters
b_mean_hp <- 0
b_mean <- rep(b_mean_hp, I)

b_var_hp <- 1000
B_cov = diag(b_var_hp, I)

# item discrimination/factor loading hyperparameters
a_mean_hp <- 0
a_mean <- rep(a_mean_hp, I)

a_var_hp <- 1000
A_cov <- diag(a_var_hp, I)

stanls_2pl_dd = list(
  "P" = P,
  "I" = I,
  "Y" = t(items_bin), 
  "b_mean" = b_mean,
  "B_cov" = B_cov,
  "a_mean" = a_mean,
  "A_cov" = A_cov
)

fit_2pl_dd <- mdl_2pl_dd$sample(
  data = stanls_2pl_dd,
  seed = 02112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  init = function() list(a = rnorm(I, mean = 5, sd = 1))
)

###############
# diagnostics #
###############

# checking convergence
fit_2pl_dd$summary()
fit_2pl_dd$cmdstan_diagnose()
fit_2pl_dd$diagnostic_summary()
max(fit_2pl_dd$summary()$rhat, na.rm = TRUE)

###################
# item parameters #
###################

# summary of the item parameters
fit_2pl_dd$summary("a") # E(Y| theta = 0)
fit_2pl_dd$summary("b") # E(Y| theta + 1) - E(Y| theta)

# extract posterior draws
draws_dd <- posterior::as_draws_rvars(fit_2pl_dd$draws())

# fixed theta values
theta_fixed <- seq(-3, 3, length.out = P)

# drawing item characteristic curves for item
draws_dd$logit <- draws_dd$mu + draws_dd$lambda * t(theta_fixed)
# ...including estimation uncertainty in theta
# draws_dd$y <- exp(draws_dd$logit) / (1 + exp(draws_dd$logit))

# comparing with other parameters estimated:
plot(x = mean(draws_dd$b), y = mean(draws_si2$b),
  xlab = "Discrimination/Difficulty Model", 
  ylab = "Slope/Intercept Model",
  main = "Difficulty Parameter EAP Estimates"
)

# comparing with other parameters estimated:
plot(
  x = mean(draws_dd$theta), y = mean(drawsSI2$theta),
  xlab = "Discrimination/Difficulty Model",
  ylab = "Slope/Intercept Model",
  main = "Difficulty Parameter EAP Estimates"
)

# comparing with other parameters estimated:
plot(
  x = sd(draws_dd$theta), y = sd(drawsSI2$theta),
  xlab = "Discrimination/Difficulty Model",
  ylab = "Slope/Intercept Model",
  main = "Theta SD Estimates"
)

########################
# Auxiliary Statistic  #        
########################

fml_2pl_dd2 <- "
data {
  int<lower=0> P;               // number of observations
  int<lower=0> I;               // number of items
  array[I, P] int<lower=0, upper=1>  Y; // item responses in an array 

  vector[I] a_mean;
  matrix[I, I] A_cov;      // prior covariance matrix for coefficients
  
  vector[I] b_mean;         // prior mean vector for coefficients
  matrix[I, I] B_cov;  // prior covariance matrix for coefficients
  
  int<lower=0> N_theta;        // n° of theta values for auxiliary statistics
  vector[N_theta] theta_fix;   // values for auxiliary statistics
}

parameters {
  vector[P] theta;          // the latent variables (one for each person)
  vector[I] a;              // the item intercepts (1/item)
  vector[I] b;               // item discriminations/loading (1/item)
}

model {
  a ~ multi_normal(a_mean, A_cov); // item discrimination/factor loadings
  b ~ multi_normal(b_mean, B_cov);             // Prior for item intercepts
  
  theta ~ normal(0, 1);    // Standardied LV, prior p.d. (mean/sd specified)
  
  for (i in 1:I){
    // Import: If we loop with '[i]' we access every person! (row major order)
    Y[i] ~ bernoulli_logit(a[i]*(theta - b[i]));
  }
}

generated quantities{
  vector[I] lambda;
  vector[I] mu;
  vector[N_theta] TCC;
  matrix[N_theta, I] item_info;
  vector[N_theta] test_info;
  
  for (v in 1:N_theta){
    TCC[v] = 0.0;
    // test info must start at -1 to include prior p.d. for theta
    test_info[v] = -1.0;  
    for (i in 1:I){
      item_info[v, i] = 0.0;
    }
  }
  
  lambda = a;
  for (i in 1:I){
    mu[i] = -1*a[i]*b[i];
    
    for (v in 1:N_theta){
      // test characteristic curve:
      TCC[v] = TCC[v] + inv_logit(a[i]*(theta_fix[v]-b[i]));
      
      // item information functions:
      item_info[v, i] = 
        item_info[v, i] + a[i]^2 * inv_logit(a[i] * (theta_fix[v] - b[i])) * 
        (1 - inv_logit(a[i] * (theta_fix[v] - b[i])));
      
      // test information functions:
      test_info[v] = test_info[v] + a[i]^2 * inv_logit(a[i] * (theta_fix[v] - b[i])) * (1 - inv_logit(a[i] * (theta_fix[v] - b[i])));
    }
  }
}
"

# Compile model
mdl_2pl_dd2 <- cmdstan_model(stan_file = write_stan_file(fml_2pl_dd2))

# values for auxiliary statistics 
theta_fix <- seq(-3, 3, length.out = P)

stanls_2pl_dd2 = list(
  "P" = P,
  "I" = I,
  "Y" = t(items_bin), 
  "b_mean" = b_mean,
  "B_cov" = B_cov,
  "a_mean" = a_mean,
  "A_cov" = A_cov,
  "N_theta" = length(theta_fix),
  "theta_fix" = theta_fix 
)

fit_2pl_dd2 <- mdl_2pl_dd2$sample(
  data = stanls_2pl_dd2,
  seed = 02112022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  # Important: Now we use a instead of lambda
  init = function() list(a = rnorm(I, mean = 5, sd = 1))
)

###############
# diagnostics #
###############

# checking cnvergence
fit_2pl_dd2$summary()
fit_2pl_dd2$cmdstan_diagnose()
fit_2pl_dd2$diagnostic_summary()
max(fit_2pl_dd2$summary()$rhat, na.rm = TRUE)

###################
# item parameters #
###################

# summary of the item parameters
fit_2pl_dd2$summary("a") # E(Y| theta = 0)
fit_2pl_dd2$summary("b") # E(Y| theta + 1) - E(Y| theta)

# extract posterior draws
draws_dd2 <- posterior::as_draws_rvars(fit_2pl_dd2$draws())

# fixed theta values
theta_fixed <- seq(-3, 3, length.out = P)

# drawing item characteristic curves for item
draws_dd2$logit <- draws_dd2$mu + draws_dd2$lambda * t(theta_fixed)
# ...including estimation uncertainty in theta
# draws_dd$y <- exp(draws_dd$logit) / (1 + exp(draws_dd$logit))

# TCC Spaghetti Plots
plot(x = theta_fix, 
   y = mean(draws_dd2$TCC),
   xlab = expression(theta), 
   ylab = "Expected Score", type = "l",
   main = "Test Characteristic Curve", lwd = 4)
tcc_arr <- draws_of(draws_dd2$TCC)
# Include uncertainty
for (d in seq_len(100)){
  lines(theta_fix, y = tcc_arr[d, ], col = "steelblue")
}
lines(theta_fix, y = mean(draws_dd2$TCC), lwd = 4)
legend(
  x = -3, y = 7,
  legend = c("Posterior Draw", "EAP"), col = c(1, 2), lty = c(1, 2), lwd = 5
)

# ICC Spaghetti Plots
itemno <- 1

plot(x = theta_fix, 
     y = mean(draws_dd2$item_info[,itemno]),
     xlab = expression(theta), 
     ylab = "Information", type = "l",
     main = paste0(itemLabel, " Information Function"), lwd = 2,
     ylim = c(0,maxInfo+.5))
item_info_arr = draws_of(draws_dd2$item_info[, itemno])
# Include uncertainty
for (d in seq_len(100)) {
  lines(x = theta_fix, y = item_info_arr[d, , ], col = "steelblue")
}
# EAP TCC
lines(
  x = theta_fix,
  y = mean(draws_dd2$item_info[, 1]),
  lwd = 7
)

# Visualize: Test Information Function
plot(
  x = theta_fix,
  y = mean(draws_dd2$test_info),
  xlab = expression(theta),
  ylab = "Information", type = "l",
  main = "Test Information Function", lwd = 2,
  ylim = c(0, 500)
)
tif_arr = draws_of(draws_dd2$test_info)
for (d in seq_len(100)) {
  lines(x = theta_fix, y = tif_arr[d, ], col = "steelblue")
}
# EAP TIF
lines(
  x = theta_fix,
  y = mean(draws_dd2$test_info),
  lwd = 7
)

# EAP TCC
plot(x = theta_fix, 
     y = apply(X=tifSamples, MARGIN=2, FUN=mean),
     xlab = expression(theta), 
     ylab = "Information", type = "l",
     main = "Test Information Function", 
     lwd = 2)

####################
# Other IRT Models #
####################

# TODO

# 1PL Model:
modelIRT_1PL_syntax <- "

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

