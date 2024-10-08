
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
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000
)

###############
# Diagnostics #
###############

# todo todo todo

# checking convergence
max(fit_cfa$summary()$rhat, na.rm = TRUE)

# item parameter results
print(fit_cfa$summary(variables = c("mu", "lambda", "psi")), n = Inf)

# showing relationship between item means and mu parameters
apply(X = conspiracy_items, MARGIN = 2, FUN = mean)

###################
# Item parameters #
###################

# define sequence of item numbers
# I_seq <- seq_len(I)
I_seq <- I

# define labels for item parameters
mu_lbl <- paste0("mu[", I_seq, "]")
lambda_lbl <- paste0("lambda[", I_seq, "]")
psi_lbl <- paste0("psi[", I_seq, "]")

# extract draws from item parameters
item_pars <- fit_cfa$draws(variables = c(mu_lbl, lambda_lbl, psi_lbl), 
  format = "draws_matrix")

# extract summary statistics for item parameters
(item_summary <- fit_cfa$summary(variables = c(mu_lbl, lambda_lbl, psi_lbl)))

# item plot
theta <- seq(-3, 3, .1) # for plotting analysis lines--x axis values

# drawing item characteristic curves for item
# E(theta|Y) 
y <- as.numeric(item_pars[1, mu_lbl]) + 
  as.numeric(item_pars[1, lambda_lbl]) * theta

plot(x = theta, y = y, type = "l", main = paste("Item", I_seq, "ICC"),
     ylim = c(-2,8), xlab = expression(theta),
     ylab = paste("Item", I_seq, "Predicted Value"))
for (draw in seq_len(50)){
  y <- as.numeric(item_pars[draw, mu_lbl]) +
        as.numeric(item_pars[draw, lambda_lbl]) * theta
  lines(x = theta, y = y)
}
# drawing limits
lines(x = c(-3, 3), y = c(5, 5), type = "l", col = 4, lwd = 5, lty = 2)
lines(x = c(-3, 3), y = c(1, 1), type = "l", col = 4, lwd = 5, lty = 2)
# drawing EAP line
y <- item_summary$mean[which(item_summary$variable == mu_lbl)] + 
  item_summary$mean[which(item_summary$variable == lambda_lbl)] * theta
lines(x = theta, y = y, lwd = 5, lty = 3, col = 2)

# multimodality (2 separate modes)
mcmc_dens(fit_cfa$draws(variables = "lambda[10]"))
# legend
legend(x = -3, y = 7, legend = c("Posterior Draw", "Item Limits", "EAP"), 
col = c(1,4,2), lty = c(1,2,3), lwd=5)


# multimodality (2 separate modes)
plot(x = item_pars[, 1], y = item_pars[, 2])
cor(x = item_pars[, 1], y = item_pars[, 2])

# investigating latent variables

#results
print(fit_cfa$summary(variables = c("theta")), n = Inf)
# Almost all parameter estimates are shrunk towards zero

# EAP distribution
hist(fit_cfa$summary(variables = c("theta"))$mean, 
main = "EAP Estimates of Theta", xlab = expression(theta))
# All the values center around zero (multimodality)

plot(density(fit_cfa$summary(variables = c("theta"))$mean), 
main = "EAP Estimates of Theta", xlab = expression(theta))
# All the values center around zero (multimodality)

# Density of All Posterior Draws
theta <- fit_cfa$draws(variables = c("theta"), format = "draws_matrix")
theta_vec <- c(theta)
hist(theta_vec)

# plotting two theta distributions side-by-side
theta1 <- "theta[1]"
theta2 <- "theta[2]"
theta_draws <- fit_cfa$draws(variables = c(theta1, theta2), format = "draws_matrix")
theta_vec <- rbind(theta_draws[,1], theta_draws[,2])
theta_df <- data.frame(observation = c(rep(theta1, nrow(theta_draws)), rep(theta2, nrow(theta_draws))), 
sample = theta_vec)
  rep(theta1, nrow(theta_draws), sample = theta_vec)
names(theta_df) <- c("observation", "sample")
ggplot(theta_df, aes(x = sample, fill = observation)) + 
  geom_density(alpha = 0.25)
# The resulting distribution is a result of combining the two
# distributions of theta[1] and theta[2]

# comparing EAP estimates with posterior SDs
plot(y = fit_cfa$summary(variables = c("theta"))$sd, 
x = fit_cfa$summary(variables = c("theta"))$mean,
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)")
# BOOM!
# under normal theory we would expect a line (constant), because
# ...but this is clearly not the case here
# we use the same standard deviation (fixed); but allowing for
# variation in a clearly nonlinear pattern emerges

## comparing EAP estimates with sum scores
plot(x = rowSums(conspiracy_items), 
y = fit_cfa$summary(variables = c("theta"))$mean,
     xlab = "Sum Score", ylab = expression(theta))

# Estimating Theta with fixed item parameters
lambda_eap <- fit_cfa$summary(variables = "lambda")$mean
mu_eap <- fit_cfa$summary(variables = "mu")$mean
psi_eap <- fit_cfa$summary(variables = "psi")$mean

# Stan syntax
fml_fixedcfa <- "

data {
  int<lower=0> P;                 // number of observations
  int<lower=0> I;               // number of items
  matrix[P, I] Y;            // item responses in a matrix

  vector[I] mu_eap;        // (fixed) EAP estimates of item intercepts
  vector[I] lambda_eap;   // (fixed) EAP estimates of item loadings
  vector[I] psi_eap;      // (fixed) EAP estimates of unique sd
}

parameters {
  vector[P] theta;                // the latent variables (one for each person)
}

model {
  
  theta ~ normal(0, 1);        // prior for latent variable (mean/sd specified)
  
  for (item in 1:I){
    Y[,item] ~ normal(mu_eap[item] + lambda_eap[item]*theta, psi_eap[item]);
  }
  
}

"

# compile model
mdl_fixedcfa <- cmdstan_model(stan_file = write_stan_file(fml_fixedcfa))

# build r list for stan
stanls_fixedcfa <- list(
  P = P,
  I = I,
  Y = conspiracy_items,
  mu_eap = mu_eap,
  lambda_eap = lambda_eap,
  psi_eap = psi_eap
)

# run MCMC chain (sample from posterior p.d.)
# note run very long chain to get a fine resolution of the tails
fit_fixedcfa <- mdl_fixedcfa$sample(
  data = stanls_fixedcfa,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 50000
)

# extracting fixed person parameter results
fixedItems_ThetaMeans <- fit_fixedcfa$summary(variables = c("theta"))$mean
fixedItems_ThetaSDs <- fit_fixedcfa$summary(variables = c("theta"))$sd

plot(y = fixedItems_ThetaSDs,
     x = fixedItems_ThetaMeans,
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)",
     ylim = c(0, 1))

# extracting person parameter (estimated) 
estimatedItems_ThetaMeans <- fit_cfa$summary(variables = c("theta"))$mean
estimatedItems_ThetaSDs <- fit_cfa$summary(variables = c("theta"))$sd

plot(y = estimatedItems_ThetaMeans,
     x = fixedItems_ThetaMeans,
     xlab = "Fixed Item Parameters", ylab = "Estimated Item Parameters", 
     main = "EAP Theta Estimates")
# Mean is almost perfectly covered, BUT...
# Mean is a stable quantitiy

plot(y = estimatedItems_ThetaSDs,
     x = fixedItems_ThetaSDs,
     xlab = "Fixed Item Parameters", ylab = "Estimated Item Parameters", 
     main = "Theta Posterior SDs")
#....the SDs are not

# With estimates fixed at the EAP values, we dramatically overstate
# our certainty in the estimates of theta, especially with a small sample size 
# Put another our uncertainty in the itemparameters does not translate to 
# uncertainty in the person parameters!

#####################
# Convergence Fails #
#####################

fit_failedcfa <- mdl_cfa$sample(
  data = stanls_cfa,
  seed = 25102022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000
)

# checking convergence
max(fit_failedcfa$summary()$rhat, na.rm = TRUE)

# item parameter results
print(fit_failedcfa$summary(variables = c("mu", "lambda", "psi")), n=Inf)

# person parameter results
print(fit_failedcfa$summary(variables = c("theta")), n = Inf)

# plotting trace
mcmc_trace(fit_failedcfa$draws(variables = "lambda"))
# Two separated modes! Three chains are stuck in one mode, one in the other

# plotting densities
mcmc_dens(fit_failedcfa$draws(variables = "lambda"))
mcmc_dens(fit_failedcfa$draws(variables = c("theta[1]", "theta[2]", "theta[3]")))
# Two separated modes! Three chains are stuck in one mode, one in the other

# investigating item parameters
itemno <- 3

mu_lbl <- paste0("mu[", itemno, "]")
lambda_lbl <- paste0("lambda[", itemno, "]")
psi_lbl <- paste0("psi[", itemno, "]")
item_pars <- failedcfa$draws(variables = c(mu_lbl, lambda_lbl, psi_lbl), 
format <- "draws_matrix")
item_summary <- failed_cfa$summary(variables = c(mu_lbl, lambda_lbl, psi_lbl))

# item plot
theta <- seq(-3, 3, .1) # for plotting analysis lines--x axis values
 
# drawing item characteristic curves for item
y <- as.numeric(item_pars[1,mu_lbl]) + as.numeric(item_pars[1,lambda_lbl]) * 
theta
plot(x = theta, y = y, type = "l", main = paste("Item", itemno, "ICC"),
     ylim = c(-2,8), xlab = expression(theta),
     ylab = paste("Item", itemno, "Predicted Value"))
for (draw in 2:nrow(item_pars)) {
  y <- as.numeric(item_pars[draw, mu_lbl]) +
  as.numeric(item_pars[draw,lambda_lbl]) * theta
  lines(x = theta, y = y)
}

# drawing limits
# ... the x-effect -- two modes, perfect reflections!
lines(x = c(-3, 3), y = c(5, 5), type = "l", col = 4, lwd = 5, lty = 2)
lines(x = c(-3, 3), y = c(1, 1), type = "l", col = 4, lwd = 5, lty = 2)

# drawing EAP line
# drawing EAP line
y <- item_summary$mean[which(item_summary$variable == mu_lbl)] + 
  item_summary$mean[which(item_summary$variable == lambda_lbl)] * theta
lines(x = theta, y = y, lwd = 5, lty = 3, col = 2)

# legend
legend(x = -3, y = 7, legend = c("Posterior Draw", "Item Limits", "EAP"), col = c(1,4,2), lty = c(1,2,3), lwd=5)

# alternative strategy for ensuring convergence to single mode of data: =========

# initial problem chains:

# checking convergence
max(fit_failedcfa$summary()$rhat, na.rm = TRUE)

# item parameter results
print(fit_failedcfa$summary(variables = c("mu", "lambda", "psi")) ,n = Inf)

#
# Stop
# Failed below!
#

# set starting values for some of the parameters
# here, we are examining what the starting values were by running a very small chain without warmup
fit_newstartcfa <- mdl_cfa$sample(
  data = stanls_cfa,
  seed = 25102022,
  chains = 1,
  parallel_chains = 1,
  iter_warmup = 0,
  iter_sampling = 10,
  # Random starting values for lambda
  init = function() list(lambda = rnorm(I, mean = 10, sd = 1)), 
  adapt_engaged = FALSE
)

modelCFA_samples2starting$draws(variables = "lambda", format = "draws_matrix")

# now we can see the sampling work (with limited warmup)
modelCFA_samples2nowarmup = modelCFA_stan$sample(
  data = modelCFA_data,
  seed = 25102022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 10,
  iter_sampling = 2000, 
  init = function() list(lambda=rnorm(nItems, mean=10, sd=2))
)

mcmc_trace(modelCFA_samples2nowarmup$draws(variables = "lambda"))

View(modelCFA_samples2nowarmup$draws(variables = "lambda", format = "draws_matrix"))

# now we can see the sampling work (with limited warmup)
modelCFA_samples2fixed = modelCFA_stan$sample(
  data = modelCFA_data,
  seed = 25102022,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000, 
  init = function() list(lambda=rnorm(nItems, mean=10, sd=2))
)

max(modelCFA_samples2fixed$summary()$rhat, na.rm = TRUE)

print(modelCFA_samples2fixed$summary(variables = c("mu", "lambda", "psi")) ,n=Inf)
print(modelCFA_samples2fixed$summary(variables = c("theta")) ,n=Inf)

plot(y = modelCFA_samples2fixed$summary(variables = c("mu", "lambda", "psi", "theta"))$mean,
     x = modelCFA_samples$summary(variables = c("mu", "lambda", "psi", "theta"))$mean,
     main = "Comparing Results from Converged", xlab = "Without Starting Values",
     ylab = "With Starting Values")
cor(modelCFA_samples2fixed$summary(variables = c("mu", "lambda", "psi", "theta"))$mean,
    modelCFA_samples$summary(variables = c("mu", "lambda", "psi", "theta"))$mean)

save.image("lecture04b.RData")
