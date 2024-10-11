
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

# Model the belief in conspiracy theories assuming a normal p.d. for the data
conspiracy_data <- read.csv("./data/conspiracies.csv")

# Use only the first 10 items
# Note: Positive values mean resemble agreement
conspiracy_items <- conspiracy_data[, 1 : 10]

# Data dimensions
P <- nrow(conspiracy_items)
I <- ncol(conspiracy_items)

# Item intercept hyperparameters
mu_mean <- rep(0, I)
Mu_cov <- diag(1000, I)

# Item discrimination/factor loading hyperparameters
lambda_mean <- rep(0, I, I)
Lambda_cov <- diag(1000, I)

# Unique standard deviation hyperparameters
psi_rate <- rep(0.01, I)

#######
# CFA #
#######

# Compile model into executable
mdl_cfa <- cmdstan_model("./stan/4b/cfa.stan", pedantic = TRUE)

# Stan list
stanls_cfa <- list(
  "P" = P,
  "I" = I,
  "Y" = conspiracy_items,
  "mu_mean" = mu_mean,
  "Mu_cov" = Mu_cov,
  "lambda_mean" = lambda_mean,
  "Lambda_cov" = Lambda_cov,
  "psi_rate" = psi_rate
)

# Fit the model to the data
fit_cfa <- mdl_cfa$sample(
  data = stanls_cfa,
  seed = 09102022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000
)

###############
# Diagnostics #
###############

# Assess convergence: summary of all parameters
fit_cfa$cmdstan_diagnose()
fit_cfa$diagnostic_summary()

# Checking convergence
max(fit_cfa$summary()$rhat, na.rm = TRUE)

###################
# Item parameters #
###################

# Summary of item intercepts
fit_cfa$summary(variables = "mu")       # E(Y|theta =0)

# Interpret the item intercept (remember: standardized LV):
# For someone with an average amount of the tendendcy to agree with 
# conspircaries, we would expect their value for item 1 to be be 2.37 out of 5
fit_cfa$summary(variables = "mu")[1,]

# Summary of factor loadings
fit_cfa$summary(variables = "lambda") # E(Y| theta + 1) - E(Y| theta)

# Interpret the factor loadings
# If we compare to respondents that differ by one (standard deviation) in their
# tendency to believe in conspiracy theory, we would expect the one with the
# heigher tendency to respond a 0.371 higher than the other
fit_cfa$summary(variables = "lambda")[1,]

# Summary of the unique standard deviations
fit_cfa$summary(variables = "psi")

# Extract posterior draws as r.v.
draws_cfa <- posterior::as_draws_rvars(fit_cfa$draws())

# Show relationship between item means and mu parameters
colMeans(conspiracy_items)

###################
# Item parameters #
###################

# Fixed theta values
theta_fixed <- seq(-3, 3, length.out = P)

# Important: I hope that recycling works properly. Since dim(psi) = 10 and 
# dim(mu+lambda*t(theta)) is 10 x 177: ino <- 13 ; draws_cfa$yhat[ino]
# (draws_cfa$mu + draws_cfa$lambda * t(draws_cfa$theta))[ino] + draws_cfa$psi[ino-10]
draws_cfa$yhat <- draws_cfa$mu + draws_cfa$lambda * t(draws_cfa$theta) +
  draws_cfa$psi
# ...including estimation uncertainty in theta
# draws_cfa$mu + draws_cfa$lambda * t(draws_cfa$theta) +  draws_cfa$psi

# Normal ICC (item characteristic curve)
itemno <- 10 
plot(NULL, ylim = c(-2,8), xlim = range(theta_fixed), xlab = expression(theta))
mu_arr <- posterior::draws_of(draws_cfa$mu[itemno])
lambda_arr <- posterior::draws_of(draws_cfa$lambda[itemno])
for (d in seq_len(2000)) {
  abline(a = mu_arr[d], b = lambda_arr[d], col = "steelblue", lwd = 0.05)
}
abline(a = mean(mu_arr[1:2000]), b = mean(lambda_arr[1:2000]), lwd = 5)
# Limits
lines(x = c(-3, 3), y = c(5, 5), type = "l", col = 2, lwd = 5, lty = 2)
lines(x = c(-3, 3), y = c(1, 1), type = "l", col = 2, lwd = 5, lty = 2)

# Multimodality (2 separate modes)
mcmc_dens(fit_cfa$draws(variables = "lambda[10]"))
plot(mu_arr, lambda_arr)
cor(mu_arr, lambda_arr)

# Investigating person parameters
summary(draws_cfa$theta)
# Almost all parameter estimates are shrunk towards zero

# Distribution of EAP estimates (posterior means)
hist(mean(draws_cfa$theta), main = "EAP Estimates of Theta", 
xlab = expression(theta))
# All the values center around zero (multimodality)

plot(density(mean(draws_cfa$theta)), main = "EAP Estimates of Theta",
xlab = expression(theta))
# All the values center around zero (multimodality)

# Density of All Posterior Draws
hist(draws_of(draws_cfa$theta))

# todo todo todo

# Plotting two theta distributions side-by-side
plot(NULL, xlim = c(-3, 3), ylim = c(0,1.5), xlab = expression(theta),
main = "EAP Estimates of Theta")
polygon(density(draws_of(draws_cfa$theta[1])), col = "slateblue1")
polygon(density(draws_of(draws_cfa$theta[2])), col = "steelblue")
# The resulting distribution is a result of combining the two
# distributions of theta[1] and theta[2]

# comparing EAP estimates with posterior SDs
plot(mean(draws_cfa$theta), sd(draws_cfa$theta),
  xlab = "E(theta|Y)",
  ylab = "SD(theta|Y)"
)
# BOOM!
# Under normal theory we would expect a line (constant), because
# ...but this is clearly not the case here!
# We use the same standard deviation (fixed); but allowing for
# variation in a clearly nonlinear pattern emerges

# todo todo todo

# Comparing EAP estimates with sum scores
plot(rowSums(conspiracy_items), mean(draws_cfa$theta),
  xlab = "Sum Score", ylab = expression(theta)
)

# Estimating Theta with fixed item parameters
lambda_eap <- mean(draws_cfa$lambda)
mu_eap <- mean(draws_cfa$mu)
psi_eap <- mean(draws_cfa$psi)


# Compile model
mdl_fixedcfa <- cmdstan_model("./stan/4b/fixedcfa.stan", pedantic = TRUE)

# Build r list for stan
stanls_fixedcfa <- list(
  "P" = P,
  "I" = I,
  "Y" = conspiracy_items,
  "mu_eap" = mu_eap,
  "lambda_eap" = lambda_eap,
  "psi_eap" = psi_eap
)

# Run MCMC chain (sample from posterior p.d.)
# note run very long chain to get a fine resolution of the tails
fit_fixedcfa <- mdl_fixedcfa$sample(
  data = stanls_fixedcfa,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 50000
)

###############
# Diagnostics #
###############

# Assess convergence: summary of all parameters
fit_fixedcfa$cmdstan_diagnose()
fit_fixedcfa$diagnostic_summary()

# Checking convergence
max(fit_fixedcfa$summary()$rhat, na.rm = TRUE)

# Extract posterior draws as r.v.
draws_fixedcfa <- posterior::as_draws_rvars(fit_fixedcfa$draws())

# Visualize person parameter (fixed) 
plot(mean(draws_fixedcfa$theta), sd(draws_fixedcfa$theta),
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)",
     ylim = c(0, 1))

# Visualize person parameter (estimated vs. fixed) 
plot(mean(draws_fixedcfa$theta), mean(draws_cfa$theta),
  xlab = "Fixed Item Parameters", ylab = "Estimated Item Parameters",
  main = "EAP Theta Estimates")
# Mean is almost perfectly covered, the mean is a stable quantitiy. BUT...

plot(sd(draws_fixedcfa$theta), sd(draws_cfa$theta),
  xlab = "Fixed Item Parameters", ylab = "Estimated Item Parameters",
  main = "Theta Posterior SDs")
#....the SDs are not

# With estimates fixed at the EAP values, we dramatically overstate our certainty in the estimates of theta, especially with a small sample size. Put another our uncertainty in the itemparameters does not translate to uncertainty in the person parameters!

#####################
# Convergence Fails #
#####################

fit_failedcfa <- mdl_cfa$sample(
  data = stanls_cfa,
  seed = 21102022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000
)

# Checking convergence
max(fit_failedcfa$summary()$rhat, na.rm = TRUE)

# Item parameter results
print(fit_failedcfa$summary(variables = c("mu", "lambda", "psi")), n = Inf)

# Person parameter results
print(fit_failedcfa$summary(variables = c("theta")), n = Inf)

# Plotting trace
mcmc_trace(fit_failedcfa$draws(variables = "lambda"))
# Two separated modes! Three chains are stuck in one mode, one in the other

# Plotting densities
mcmc_dens(fit_failedcfa$draws(variables = "lambda"))
mcmc_dens(fit_failedcfa$draws(variables = c("theta[1]", "theta[2]", "theta[3]")))
# Two separated modes! Three chains are stuck in one mode, one in the other

# Extract posterior draws
draws_failedcfa <- posterior::as_draws_rvars(fit_failedcfa$draws())

# Visualize the problem of two modes
theta_fixed <- seq(-3, 3, length.out = P)
itemno <- 10 
plot(NULL, ylim = c(-2, 8), xlim = range(theta_fixed), xlab = expression(theta))
mu_arr <- posterior::draws_of(draws_failedcfa$mu[itemno])
lambda_arr <- posterior::draws_of(draws_failedcfa$lambda[itemno])
for (d in seq_len(4000)) {
  abline(a = mu_arr[d], b = lambda_arr[d], col = "steelblue", lwd = 0.05)
}
# Multimodality
abline(a = mean(mu_arr), b = mean(lambda_arr), col = 2, lwd = 5)
abline(a = mean(mu_arr[1:2000]), b = mean(lambda_arr[1:2000]), lwd = 5)
abline(a = mean(mu_arr[2001:4000]), b = mean(lambda_arr[2001:4000]), lwd = 5)
# Limits
lines(x = c(-3, 3), y = c(5, 5), type = "l", col = 5, lwd = 5, lty = 2)
lines(x = c(-3, 3), y = c(1, 1), type = "l", col = 5, lwd = 5, lty = 2)

# Multimodality (2 separate modes)
mcmc_dens(fit_failedcfa$draws(variables = "lambda[10]"))
plot(mu_arr, lambda_arr)
cor(mu_arr, lambda_arr)

#########################################
# Ensuring convergence to a single mode #
#########################################

# checking convergence
max(fit_failedcfa$summary()$rhat, na.rm = TRUE)

# item parameter results
print(fit_failedcfa$summary(variables = c("mu", "lambda", "psi")), n = Inf)

# Set starting values for some of the parameters
# here, we are examining what the starting values were by running a very small chain without warmup
fit_cfastartval <- mdl_cfa$sample(
  data = stanls_cfa,
  seed = 25102022,
  chains = 1,
  iter_warmup = 0,
  iter_sampling = 10,
  # Random starting values for lambda
  init = function() list("lambda" = rnorm(I, mean = 10, sd = 1)),
  adapt_engaged = FALSE
)

# Now we can see the sampling work (with limited warmup)
fit_cfanowarmup <- mdl_cfa$sample(
  data = stanls_cfa,
  seed = 25102022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 10,
  iter_sampling = 2000, 
  init = function() list("lambda" = rnorm(I, mean = 10, sd = 2))
)

# See if first samples start at values of 10, since we set 'mean = 10'
View(fit_cfanowarmup$draws(variables = "lambda", format = "draws_matrix"))

# Traceplots
mcmc_trace(fit_cfanowarmup$draws(variables = "lambda"))

# Now we can see the sampling work (with limited warmup)
fit_warmcfa <- mdl_cfa$sample(
  data = stanls_cfa,
  seed = 25102022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000, 
  init = function() list(lambda = rnorm(I, mean = 10, sd = 2))
)

###############
# Diagnostics #
###############

# Assess convergence: summary of all parameters
fit_warmcfa$cmdstan_diagnose()
fit_warmcfa$diagnostic_summary()

# Checking convergence
max(fit_warmcfa$summary()$rhat, na.rm = TRUE)

# Extract posterior draws
draws_warmcfa <- posterior::as_draws_rvars(fit_warmcfa$draws())

# Visualize comparison between failed and not failed models
plot(NULL, xlim = c(-3,3), ylim = c(-3, 3))
points(mean(draws_warmcfa$mu), mean(draws_failedcfa$mu), col = 2)
points(mean(draws_warmcfa$lambda), mean(draws_failedcfa$lambda),
  col = 3, cex = 4)
points(mean(draws_warmcfa$psi), mean(draws_failedcfa$psi), col = 4)
points(mean(draws_warmcfa$theta), mean(draws_failedcfa$theta), col = 1)
abline(a = 0, b = 1, lty = 2)
# Lambda and theta are most problematic!