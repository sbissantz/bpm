
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

# Number of dimensions
D <- 2
# Test if the data are conformable with two factors: 
# (1) A government factor & (2) a non-goverment factor 

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

###########################################
# Multidimensional 2POL with marker items #
# MD-GRM                                  #
###########################################

# Compile model
mdl_md2polsim <- cmdstan_model("./stan/4h/md2pol_si_marker.stan",
  pedantic = TRUE)

############
# Q-Matrix #
############

# Initialize a I X D matrix with zeros
Q <- matrix(0, I, D)
colnames(Q) <- c("Gov", "NonGov")
rownames(Q) <- paste0("i", seq_len(I))
# Determine which items measure each factor
# Factor 1 is measured by (loads on) items...
itms_d1 <- c(2, 5, 7, 8, 9)
Q[itms_d1, 1] <- 1
# Factor 2 is measured by (loads on) items...
itms_d2 <- c(1, 3, 4, 6, 10)
Q[itms_d2, 2] <- 1

# Import: We have no multidimensional items, every factor is measured by a
# unique set of items.
Q

# Item threshold hyperparameter
Thr_mean <- replicate(C - 1, rep(0, I)) # 10 x 4
THR_cov <- array(0, dim = c(10, 4, 4)) # 10 x 4 x 4
for(d in seq_len(I)) {
  THR_cov[d , ,] <- diag(10, C - 1)
}

# Item discrimination/factor loading hyperparameters
# Note: I - D because of marker items
lambda_mean <- rep(0, I - D)
Lambda_cov <- diag(10, I - D)

# Latent trait hyperparameters
Theta_mean <- rep(0, D)
theta_sd_loc <- rep(0, D)
theta_sd_scl <- rep(0.5, D)

#############
# Stan list #
#############

stanls_md2polsim <- list(
  "P" = P,
  "I" = I,
  "C" = C,
  "D" = D,
  "Q" = Q,
  # Important transpose (array in stan are in row major order)
  "Y" = t(citems),
  "thr_mean" = Thr_mean,
  "Thr_cov" = THR_cov,
  "lambda_mean" = lambda_mean,
  "Lambda_cov" = Lambda_cov,
  "theta_mean" = Theta_mean,
  "theta_sd_loc" = theta_sd_loc,
  "theta_sd_scl" = theta_sd_scl
)

# Multiple standardized sum scores to start theta in the correct location
lambda_init <- rnorm(I, mean = 5, sd = 1)
sum_scores <- as.matrix(citems) %*% Q

# Standardize sum scores (mean: 0, sd: 1)
theta_init <- scale(sum_scores)

# Set any missing values to a random number
na_pat <- which(is.na(thetaInit))
theta_init[na_pat] <- rnorm(1, 0, 1)

# Run MCMC chain (sample from posterior p.d.)
fit_md2polsim <- mdl_md2polsim$sample(
  data = stanls_md2polsim,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  # Mean should be below 10, since the log of it is too large
  init = function() list("lambda" = lambda_init, "theta" = theta_init)
)

##############
# Diagnostic #
##############

# Checking convergence
fit_md2polsim$cmdstan_diagnose()
fit_md2polsim$diagnostic_summary()
max(fit_md2polsim$summary()$rhat, na.rm = TRUE)

# item parameter results
print(
  fit_md2polsim$summary(
    variables = c("theta_sd", "Theta_cov", "Theta_cor", "Lambda", "mu")
  ), n = Inf)
