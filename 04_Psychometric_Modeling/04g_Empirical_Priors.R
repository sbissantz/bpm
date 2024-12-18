
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
mdl_cfa <- cmdstan_model("./stan/4b/cfa.stan", pedantic = TRUE)

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
     main = "Uninformative Mu Prior - Empirical Mu Prior")
# 9/10 times the uniformative prior p.d.'s EAP(mu) is higher 
sum(mean(draws_cfa$mu) > mean(draws_cfappip$mu))

# Comparing intercept SD estimates: uninformative vs. empirical prior

plot(sd(draws_cfa$mu), sd(draws_cfappip$mu),
     xlab = "Uninformative Prior", ylab = "Empirical Prior", 
     main = "Comparing SDs for Mu")

hist(sd(draws_cfa$mu) - sd(draws_cfappip$mu),
     xlab = "Mu SD Difference",
     main = "Uninformative Mu Prior - Empirical Mu Prior")
# 7/10 times the uniformative prior p.d.'s SD(mu) is higher 
sum(sd(draws_cfa$mu) > sd(draws_cfappip$mu))

# Comparing factor loading EAP estimates: uninformative vs. empirical prior

plot(mean(draws_cfa$lambda), mean(draws_cfappip$lambda),
     xlab = "Uninformative Prior", ylab = "Empirical Prior",
     main = "Comparing EAPs for Lambda")

hist(mean(draws_cfa$lambda) - mean(draws_cfappip$lambda),
     xlab = "Lambda EAP Difference", 
     main = "Uninformative Lambda Prior - Empirical Lambda Prior")
# 7/10 times the uniformative prior p.d.'s EAP(lambda) is higher 
sum(mean(draws_cfa$lambda) > mean(draws_cfappip$lambda))

# Comparing factor loading SD estimates: uninformative vs. empirical prior

plot(sd(draws_cfa$lambda), sd(draws_cfappip$lambda),
     xlab = "Uninformative Prior", ylab = "Empirical Prior", 
     main = "Comparing SDs for Lambda")

hist(sd(draws_cfa$lambda) - sd(draws_cfappip$lambda),
     xlab = "Lambda SD Difference", 
     main = "Uninformative Prior - Empirical Lambda Prior")
# 7/10 times the uniformative prior p.d.'s SD(lambda) is higher
sum(sd(draws_cfa$lambda) > sd(draws_cfappip$lambda))

# Comparing unique SD EAP estimates: uninformative vs. empirical prior

plot(mean(draws_cfa$psi), mean(draws_cfappip$psi),
     xlab = "Uninformative Prior", ylab = "Empirical Prior", 
     main = "Comparing EAPs for Psi")
hist(mean(draws_cfa$psi) - mean(draws_cfappip$psi),
     xlab = "Psi EAP Difference",
     main = "Uninformative Prior - Empirical Psi Prior")
# 9/10 times the uniformative prior p.d.'s EAP(psi) is higher
sum(mean(draws_cfa$psi) > mean(draws_cfappip$psi))

# Comparing theta EAP estimates: uninformative vs. empirical prior

plot(mean(draws_cfa$theta), mean(draws_cfappip$theta),
     xlab = "Uninformative Prior", ylab = "Empirical Prior",
     main = "Comparing EAPs for Mu")

hist(mean(draws_cfa$theta) - mean(draws_cfappip$theta),
     xlab = "Theta EAP Difference", 
     main = "Uninformative Prior - Empirical Prior")
# 42/177 times the uniformative prior p.d.'s EAP(mu) is higher 
sum(mean(draws_cfa$theta) > mean(draws_cfappip$theta))

# Comparing theta SD estimates: uninformative vs. empirical prior

plot(sd(draws_cfa$theta), sd(draws_cfappip$theta),
     xlab = "Uninformative Prior", ylab = "Empirical Prior",
     main = "Comparing EAPs for Mu")

hist(sd(draws_cfa$theta) - sd(draws_cfappip$theta),
     xlab = "Theta SD Difference", 
     main = "Uninformative Prior - Empirical Prior")
# 11/177 times the uniformative prior p.d.'s SD(mu) is higher 
sum(sd(draws_cfa$theta) > sd(draws_cfappip$theta))

#############################
# CFA with empirical priors #
#############################
# 2. Empirical prior on item parameters and theta

# Import: Do not use an empirical prior on theta!

# Compile model into executable
mdl_cfapp <- cmdstan_model("./stan/4g/cfa_pp.stan", pedantic = TRUE)

#############
# Stan list # 
#############

stanls_cfapp <- list(
  "P" = P,
  "I" = I,
  "Y" = citems,
  "mu_hypmean" = 0,
  "mu_hypsd" = 1,
  "mu_hyprate" = 0.1,
  "lambda_hypmean" = 0,
  "lambda_hypsd" = 1,
  "lambda_hyprate" = 0.1,
  "psi_hyprate" = 0.1,
  "theta_hypmean" = 0,
  "theta_hypsd" = 1,
  "theta_hyprate" = 0.1
)

# Fit the model to the data
fit_cfapp <- mdl_cfapp$sample(
  data = stanls_cfapp,
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
fit_cfapp$cmdstan_diagnose()
# Divergent transitions – reparameterize the model!
fit_cfapp$diagnostic_summary()

# Checking convergence
max(fit_cfapp$summary()$rhat, na.rm = TRUE)

###########
# Summary #
###########

print(fit_cfapp$summary(c("mu", "mu_mean", "mu_sd", "lambda", "lambda_mean", "lambda_sd", "psi", "psi_rate", "theta", "theta_mean", "theta_sd")), n = Inf)

#########
# Draws #
#########

draws_cfapp <- posterior::as_draws_rvars(fit_cfapp$draws())

# Trace plots
mcmc_trace(fit_cfapp$draws(variables = c("theta_mean", "theta_sd")))
# Important: The results are so bad, because to estimate the variance of theta
# we need a marker item. Here, we try to get the posterior p.d. only on the 
# basis of the prior p.d.. This identification is too weak! We need stronger
# identification – we need the data likelihood to be identified.

#############################
# CFA with empirical priors #
#############################
# 3. Empirical prior on theta, fixed item parameters

# Compile model into executable
mdl_cfappfixip <- cmdstan_model("./stan/4g/cfa_ppfixip.stan", pedantic = TRUE)

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

stanls_cfappfixip <- list(
  "P" = P,
  "I" = I,
  "Y" = citems,
  "mu_mean" = mu_mean,
  "Mu_cov" = Mu_cov,
  "lambda_mean" = lambda_mean,
  "Lambda_cov" = Lambda_cov,
  "psi_rate" = psi_rate,
  "theta_hypmean" = 0,
  "theta_hypsd" = 1,
  "theta_hyprate" = 0.1
)

# Fit the model to the data
fit_cfappfixip <- mdl_cfappfixip$sample(
  data = stanls_cfappfixip,
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
fit_cfappfixip$cmdstan_diagnose()
# Divergent transitions – reparameterize the model!
fit_cfappfixip$diagnostic_summary()

# Checking convergence
max(fit_cfapp$summary()$rhat, na.rm = TRUE)

###########
# Summary #
###########

print(fit_cfappfixip$summary(c("mu", "lambda", "psi", "theta", "theta_mean", "theta_sd")), n = Inf)

#########
# Draws #
#########

draws_cfappfixip <- posterior::as_draws_rvars(fit_cfappfixip$draws())

# Trace plots
mcmc_trace(fit_cfappfixip$draws(variables = c("theta_mean", "theta_sd")))
# Important: Very BAD! Do not touch theta this way!
