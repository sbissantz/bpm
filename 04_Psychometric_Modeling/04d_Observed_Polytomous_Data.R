
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

# Item means
# Note: the mean is the proportion of respondents who agreed with the item
colMeans(citems)

#######
# CFA #
#######

# Compile model
mdl_cfa <- cmdstan_model("./stan/4b/cfa.stan", pedantic = TRUE)

# Item intercept hyperparameters
mu_mean <- rep(0, I)
Mu_cov <- diag(1000, I)

# Item discrimination/factor loading hyperparameters
lambda_mean <- rep(0, I, I)
Lambda_cov <- diag(1000, I)

# Unique standard deviation hyperparameters
psi_rate <- rep(0.01, I)

# Stan list
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

# Fit model to the data
fit_cfa <- mdl_cfa$sample(
  data = stanls_cfa,
  seed = 112,
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
fit_cfa$cmdstan_diagnose()
fit_cfa$diagnostic_summary()

# Checking convergence
max(fit_cfa$summary()$rhat, na.rm = TRUE)

# item parameter results
print(fit_cfa$summary(variables = c("mu", "lambda", "psi")), n = Inf)

# Extract posterior draws as r.v.
draws_cfa <- posterior::as_draws_rvars(fit_cfa$draws())

###################
# Item parameters #
###################

# Fixed theta values
theta_fixed <- seq(-3, 3, length.out = P)

# Normal ICC
itemno <- 10
plot(NULL, ylim = c(-2, 6), xlim = range(theta_fixed), xlab = expression(theta))
mu_arr <- posterior::draws_of(draws_cfa$mu[itemno])
lambda_arr <- posterior::draws_of(draws_cfa$lambda[itemno])
for (d in seq_len(2000)) {
  abline(a = mu_arr[d], b = lambda_arr[d], col = "steelblue", lwd = 0.05)
}
abline(a = mean(mu_arr[1:2000]), b = mean(lambda_arr[1:2000]), lwd = 5)
# Limits
lines(x = c(-3, 3), y = c(5, 5), type = "l", col = 2, lwd = 5, lty = 2)
lines(x = c(-3, 3), y = c(1, 1), type = "l", col = 2, lwd = 5, lty = 2)

# Alternative way for visualization I
draws_cfa$y <- draws_cfa$mu + draws_cfa$lambda * t(theta_fixed)
itemno <- 10
plot(NULL,
  main = paste("Item", itemno, "ICC"),
  ylim = c(-2, 6), xlim = range(theta_fixed),
  xlab = expression(theta))
yno_arr <- posterior::draws_of(draws_cfa$y[itemno, ])
for (d in 1:100) {
  lines(theta_fixed, yno_arr[d, 1, ], col = "steelblue", lwd = 0.5)
}
lines(theta_fixed, mean(draws_cfa$y[itemno, ]), lwd = 5)
legend("topleft",
  legend = c("Posterior Draw", "EAP"),
  col = c("steelblue", "black"), lty = c(1, 1), lwd = 5
)
# Limits
lines(x = c(-3, 3), y = c(5, 5), type = "l", col = 2, lwd = 5, lty = 2)
lines(x = c(-3, 3), y = c(1, 1), type = "l", col = 2, lwd = 5, lty = 2)

# Alternative way for visualization II
itemno <- 10
draws_cfa$y10 <- draws_cfa$mu[itemno] + draws_cfa$lambda[itemno] * theta_fixed
plot(NULL,
  main = paste("Item", itemno, "ICC"),
  ylim = c(-2, 6), xlim = range(theta_fixed),
  xlab = expression(theta))
yno_arr <- posterior::draws_of(draws_cfa$y10)
for (d in 1:100) {
  lines(theta_fixed, yno_arr[d,], col = "steelblue", lwd = 0.5)
}
lines(theta_fixed, mean(draws_cfa$y[itemno, ]), lwd = 5)
legend("topleft",
  legend = c("Posterior Draw", "EAP"),
  col = c("steelblue", "black"), lty = c(1, 1), lwd = 5
)
# Limits
lines(x = c(-3, 3), y = c(5, 5), type = "l", col = 2, lwd = 5, lty = 2)
lines(x = c(-3, 3), y = c(1, 1), type = "l", col = 2, lwd = 5, lty = 2)

#######################################
# 2 PL SI with Binomial(!) Likelihood #
# (Slope-Intercept Form)              # 
#######################################

O <- 5 # Number of options (1, 2, 3, 4, 5)
T <- 4 # Number of trials (0, 1, 2, 3, 4)

# Idea: Use a binomial likelihood for polytomous data
# Item responses are the n° successes in n° trials.
# E.g. if a person selects "3" out of 5 categories, we have 3 successes.
# However the binomial starts at zero, so we recode item_repsponse - 1
# Therefore if a person chooses a 3/5 on the original scale there are 3-1 = 2
# successes in 4 trial. The n° trials is constant (across all r.e.s).
# Problem: Assumes that responses are unimodal.

# Data must start at zero (orginal scale: 1-5)
# Subtract minus 1 from each value (recycling)
citems_binom <- citems - 1
head(citems_binom)

# Check first item
itemno <- 1 
table(citems_binom[, itemno])

# Determine maximum value for each item
(N <- apply(citems_binom, 2, max))

# Compile model
mdl_binom2pl_si <- cmdstan_model("./stan/4d/binom2pl_si.stan", pedantic = TRUE)

# Item intercept hyperparameters
mu_mean <- rep(0, I)
Mu_cov <- diag(1000, I)

# item discrimination/factor loading hyperparameters
lambda_mean <- rep(0, I)
Lambda_cov <- diag(1000, I)

#############
# Stan list #
#############

stanls_binom2pl_si <- list(
  "P" = P,
  "I" = I,
  "N" = N,
  "Y" = t(citems_binom),  # Transpose – Stan: arrays in row major order
  "mu_mean" = mu_mean,
  "Mu_cov" = Mu_cov,
  "lambda_mean" = lambda_mean,
  "Lambda_cov" =  Lambda_cov
)

# Fit model
fit_binom2plsi <- mdl_binom2pl_si$sample(
  data = stanls_binom2pl_si,
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
fit_binom2plsi$cmdstan_diagnose()
fit_binom2plsi$diagnostic_summary()

# Checking convergence
max(fit_binom2pl_si$summary()$rhat, na.rm = TRUE)

##########################
# Item parameter results #
##########################

print(fit_binom2pl_si$summary(variables = c("mu", "lambda")), n = Inf)

# E(Y| theta = theta_bar): A person with an average belief in conspiracies has a log-odds of -0.844 to succeed in any of the four trials, that is say yes to any one option 0,...,4. 
fit_binom2plsi$summary(variables = "mu")[1,] # -0.844

# How to make predictions on the outcome scale?
# Translate the log odds into a probability
plogis(-0.844) # 0.3
# Note: The mean of a bernoulli r.v. is E[X] = n*p.
# n = 4 (0,...,4), p = 0.3; therefore 0.3 * 4 = 1.2
# Note: The outcome scale is (1,...,5), therefore: 1.2 + 1 = 2.2.
# Therefore we expect someone with an average belief in conspircacies to respond
# 2.2 on the original scale; that is somewhere between disagree & neutral 

#########
# Draws #
#########

# Extract posterior draws
draws_binom2plsi <- posterior::as_draws_rvars(fit_binom2plsi$draws())

# Fixed theta values
theta_fixed <- seq(-3, 3, length.out = P)

draws_binom2plsi$logodds <- with(
  draws_binom2plsi, t(mu + lambda * t(theta_fixed)) # t() again to make P X I
)
# Include estimation uncertainty in theta
#draws_binom2pl_si$logodds <- with(
  #draws_binom2pl_si, t(mu + lambda * t(theta))
#)

draws_binom2plsi$prob <- with(
  # Convert to probabilit (logit not implemented in rvars)
  draws_binom2plsi, exp(logodds) / (1 + exp(logodds))
)

# Visualize OCCs
# Option characteristic curves
itemno <- 10
plot(c(-3, 3), c(0, 1), type = "n", main = "Option Characteristic Curve",
      xlab = expression(theta), ylab = "P(Y |theta)")
prob10_arr <- draws_of(draws_binom2plsi$prob[, itemno])
for (o in seq_len(O)) { # For each of the five options
  for (d in 1:100) {    # For each draw
    # lines(theta_fixed, dbinom(1, T, prob = prob10_arr[d, , ]))
    lines(theta_fixed, dbinom(o - 1, T, prob = prob10_arr[d, , ]),
          col = o, lwd = 0.3
    )
  }
  # EAP lineas
  lines(theta_fixed, dbinom(o - 1, T,
        prob = mean(draws_binom2plsi$prob[, itemno])), lwd = 4)
}
legend("left", legend = paste("Option", 1:5), col = 1:5, lwd = 3)

#################################
# Investigating item parameters #
#################################

itemno <- 10
# Make predictions on the original scale. 
# The mean of a bernoulli r.v. is E[X] = n * p.
# (a) This is the probability times the number of trials (size = 4) which gives # values between 0 and 4. (b) But since the original scale is between 1 and 5, # we also need to add 1.
# Given: n = 4, p, +1 to translate back on the outcome scale 
draws_binom2plsi$y <- 4 * draws_binom2plsi$prob + 1

# Binomial ICC (item characteristic curve) for item 5
# i.e.: E(Y_10| theta) -- on the outcome scale (1...5)
plot(theta_fixed, mean(draws_binom2plsi$y[, itemno]), type = "l",
  main = paste("Item", itemno, "ICC"), ylim = c(1, 5), lwd = 2,
  xlab = expression(theta),
  ylab = paste("Item", itemno, "Retrodicted Value")
)
yno_arr <- posterior::draws_of(draws_binom2plsi$y[, itemno])
for (d in 1:100) {
  lines(theta_fixed, yno_arr[d, , 1], col = "steelblue", lwd = 0.5)
}
# EAP
lines(theta_fixed, mean(draws_binom2plsi$y[, itemno]), lwd = 5)
# Drawing limits
lines(x = c(-3,3), y = c(5,5), type = "l", col = 4, lwd=5, lty=2)
lines(x = c(-3,3), y = c(1,1), type = "l", col = 4, lwd=5, lty=2)
# Legend
legend(-3, 4,
  legend = c("Posterior Draw", "EAP"),
  col = c("steelblue", "black"), lty = c(1, 1), lwd = 5
)

# EAP Estimates of Latent Variables
hist(mean(draws_binom2plsi$theta), main = "EAP Estimates of Theta")

# Comparing Two Posterior Distributions
plot(c(-1,2), c(0,2.2), type = "n")
theta1_arr <- as_draws_array(draws_binom2plsi$theta[1])
polygon(density(theta1_arr), col = "2")
theta2_arr <- as_draws_array(draws_binom2plsi$theta[2])
polygon(density(theta2_arr), col = "3")

# Comparing Latent Variable Posterior Means and SDs
# Comparing EAP Estimates with Posterior SDs
# Note: This is the inverse of the Information function 
plot(mean(draws_binom2plsi$theta),
    sd(draws_binom2plsi$theta),
    xlab = "E(theta|Y)", ylab = "SD(theta|Y)", main = "Mean vs SD of Theta")

# Comparing EAP Estimates with Sum Scores
# Nonlinear pattern because of the logit vs id link
plot(mean(draws_binom2plsi$theta), rowSums(citems),
     ylab = "Sum Score", xlab = expression(theta))

# Comparing Thetas: Binomial vs Normal
# Nonlinear pattern. Important: Max of Binomial around 2, Max of CFA around 3
# We get different(!) estimates for the LV, we order people differently.
# Binomial puts people 1/2 SD lower than the CFA model.
# "Sorry you are 1/2 SD less intelligent because I picked the wrong p.d."
plot(mean(draws_binom2plsi$theta), mean(draws_cfa$theta),
    ylab = "Normal", xlab = "Binomial")
# Draws into question the use of factor analytic models for Likert data!

############################
# 2 POL SI (ordered-logit) #
# (Slope-Intercept Form)   # 
############################

# compile model
mdl_2polsi <- cmdstan_model("./stan/4d/2pol_si.stan", pedantic = TRUE)

# Number of response categories
# Important: Data need successive integers from 1 to highest number
# (Recode if not consistent)
C <- 5

# Item intercept hyperparameters
Thr_mean <- replicate(C - 1, rep(0, I)) # 10 x 4
THR_cov <- array(0, dim = c(10, 4, 4)) # 10 x 4 x 4
for(d in seq_len(I)) {
  THR_cov[d , ,] <- diag(1000, C - 1)
}

# Item discrimination/factor loading hyperparameters
lambda_mean <- rep(0, I)
Lambda_cov <- diag(1000, I)

#############
# Stan list #
#############

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

# Run MCMC chain (sample from posterior p.d.)
fit_2polsi <- mdl_2polsi$sample(
  data = stanls_2polsi,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  # Mean should be below 10, since the log of it is too large
  init = function() list(lambda = rnorm(I, mean = 5, sd = 1))
)

###############
# Diagnostics #
###############

# Checking convergence
fit_2polsi$cmdstan_diagnose()
fit_2polsi$diagnostic_summary()
max(fit_2polsi$summary()$rhat, na.rm = TRUE)

# EES & overall
print(fit_2polsi$summary(), n = Inf)

#########
# Draws #
#########

draws_2polsi <- posterior::as_draws_rvars(fit_2polsi$draws())

###################
# Item parameters #
###################

# Item parameters
print(fit_2polsi$summary(variables = c("lambda", "mu")), n = Inf)

# Visualize the OCC for item number 10
itemno <- 10

# Fixed theta values
theta_fixed <- seq(-3, 3, length.out = P)

# Empty container for the category probabilities of item 10
draws_2polsi$pno <- rvar(array(0, dim = c(8000, P, C)))

# Calculate the category probabilities
for (c in seq_len(C)) {
  if (c == 1) {
    draws_2polsi$pno[, c] <- with(draws_2polsi,
      1 - (1 / (1 + exp(-(mu[itemno, c] + lambda[itemno] * theta_fixed)))))
  } else if (c == 5) {
    draws_2polsi$pno[, c] <- with(draws_2polsi,
      (1 / (1 + exp(-(mu[itemno, c-1] + lambda[itemno] * theta_fixed)))) + 0)
  } else {
    draws_2polsi$pno[, c] <- with(draws_2polsi,
      1 / (1 + exp(-(mu[itemno, c - 1] + lambda[itemno] * theta_fixed))) -
        1 / (1 + exp(-(mu[itemno, c] + lambda[itemno] * theta_fixed))))
  }
}
# Visualize (with uncertainty)
plot(c(-3, 3), c(0, 1), type = "n", main = "Option Characteristic Curve",
  xlab = expression(theta), ylab = "P(Y |theta)"
)
for (c in seq_len(C)) {
  for (d in seq_len(100)) {
    p_arr <- draws_of(draws_2polsi$pno[, c])
    lines(theta_fixed, p_arr[d, , ], lwd = 0.4, lty = c, col = c + 1)
  }
  # EAP lines
  lines(theta_fixed, mean(draws_2polsi$pno[, c]), lty = c, lwd = 5)
}
legend("left", legend = paste("Category", 1:5), col = 2:6, lty = 1:5, lwd = 3)
# The OCCS of the Binomial are way more symmetric than those from the GRM.
# This is because the Binomial uses the p.m.f. to get the probabilities for each
# category. In the GRM the probabilities are governed by from the *estimated*
# intercepts. In the GRM we have 4 submodels (p1,...,p4) with different
# intercepts (but parallel slopes: 1 – no parameter like in 3PL) therefore
# their location differ – some are further apart than others.

# Make predictions on the outcome scale (1...5)
# Expected value for the item number 10
expctitmno <- with(
  draws_2polsi,
  # Probability weight * option/category
  pno[, 1] * 1 + pno[, 2] * 2 + pno[, 3] * 3 + pno[, 4] * 4 + pno[, 5] * 5
  # Note: This gives us values between 1...5 
)
# Extendable version 
expctitmno <- 0
for (c in seq_len(C)) {
  expctitmno <- expctitmno + draws_2polsi$pno[, c] * c
}
expctitmno_arr <- draws_of(expctitmno)
plot( c(-3,3), c(1,5), type = "n", xlab = expression(theta), ylab = paste("Item", itemno,"Expected Value"))
for(d in 1:100) {
  lines( theta_fixed, expctitmno_arr[d, ,], col = "steelblue", lwd = 0.4)
}
# EAP line
lines( theta_fixed, mean(expctitmno), lwd = 5)
# Boundaries (1...5)
lines(x = c(-3, 3), y = c(5, 5), type = "l", col = 2, lwd = 2, lty = 2)
lines(x = c(-3, 3), y = c(1, 1), type = "l", col = 2, lwd = 2, lty = 2)
# Interpret
abline(v = c(0, 1, 2), lty = 2)
# S.o. with an average belief in conspiracy is expected to answer 1 on item 10
# S.o. with a high belief in conspiracy (theta = 2) is expecte to answer 4 on
# item 10.

# EAP Estimates of Latent Variables
hist(mean(draws_2polsi$theta), main = "EAP Estimates of Theta", xlab = expression(theta), probability = TRUE)
# Scale to probabilites and density
hist(mean(draws_2polsi$theta), main = "EAP Estimates of Theta", xlab = expression(theta), probability = TRUE)
lines(density(mean(draws_2polsi$theta)), lwd = 2)
# Instead of the binomial model the GRM allows for multimodality, because we
# allow for a shoft in location.

# Comparing Two Posterior Distributions
plot(c(-1,2), c(0,2.2), type = "n", xlab = "sample", ylab = "density")
theta1_arr <- as_draws_array(draws_2polsi$theta[1])
polygon(density(theta1_arr), col = "2")
theta2_arr <- as_draws_array(draws_2polsi$theta[2])
polygon(density(theta2_arr), col = "3")

# Comparing EAP Estimates with Posterior SDs
# Inverse item information plot
plot(mean(draws_2polsi$theta), sd(draws_2polsi$theta),
  xlab = "E(theta|Y)", ylab = "SD(theta|Y)", main = "Mean vs SD of Theta")
# Where is theta most precise? Multimodality! Information functions in 
# polytomous items are not always single-peaked. So you can have multiple spots
# of maximum information.

# Comparing EAP Estimates with Sum Scores
plot(mean(draws_2polsi$theta), rowSums(citems),
  ylab = "Sum Score", xlab = expression(theta))
# Non-linear function because of logistic function / logit link

# Comparing Thetas: Ordered Logit vs Normal:
plot(mean(draws_2polsi$theta), mean(draws_cfa$theta),
  ylab = "Normal", xlab = "Ordered Logit")
# Non-linear function because of logistic function / logit link

# Comparing Theta SDs: Ordered Logit vs Normal:
plot(sd(draws_2polsi$theta), sd(draws_cfa$theta),
  main = "Posterior SDs",
  ylab = "Normal", xlab = "Ordered Logit")

# Which is bigger?
hist(sd(draws_2polsi$theta) - sd(draws_cfa$theta),
  main = "SD(normal) - SD(ordered)")
abline(v = 0, lwd = 2, lty = 2)

# If we compare the info functions between 2POL and CFA the curves of the CFA
# model are úsually much higher.

# How to get more precision/reliability and more precision for theta?
# 1. More items
# 2. More categories (max. continous – CFA)
# More categories allow for more info
# Problem: People cannot differentiate above a certain thresholds.

# Comparing Thetas: Ordered Logit vs Binomial:
plot(mean(draws_2polsi$theta), mean(draws_binom2plsi$theta),
  ylab = "Binomial", xlab = "Ordered Logit")
# Little non-linearity. Which one should we take?
# The one that fits the data best / makes the best OOS predictions

# Comparing Theta SDs: Ordered Logit vs Binomial:
hist(sd(draws_binom2plsi$theta) - sd(draws_2polsi$theta),
  main = "SD(binomial) - SD(ordered)",
  ylab = "Normal", xlab = "Ordered Logit")
# Binomial


# Which is bigger?
hist(modelBinomial_samples$summary(variables = c("theta"))$sd-
       modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     main = "SD(binomial) - SD(ordered)")

############################################
# 2 PCL SI (categorical/multinomial-logit) #
# (Slope-Intercept Form)                   #
# aka. Nominal Response Model – in IRT     #
############################################

# todo todo todo

conspiracyItems = citems

# Compile model
mdl_2pclsi <- cmdstan_model("./stan/4d/2pcl_si.stan", pedantic = TRUE)

# item threshold hyperparameters
mu_mean <- rep(0, I)
Mu_mean <- replicate(C - 1, mu_mean) 

MU_cov <- replicate(C - 1, rep(0, I)) # 10 x 4
MU_cov <- array(0, dim = c(10, 4, 4)) # 10 x 4 x 4
for(d in seq_len(I)) {
  MU_cov[d , ,] <- diag(1, C - 1)
}

lambda_mean <- rep(0, I)
Lambda_mean <- replicate(C - 1, mu_mean) 

LAMBDA_cov <- replicate(C - 1, rep(0, I)) # 10 x 4
LAMBDA_cov <- array(0, dim = c(10, 4, 4)) # 10 x 4 x 4
for(d in seq_len(I)) {
  LAMBDA_cov[d , ,] <- diag(1, C - 1)
}

# todo todo todo: Why?

stanls_2pclsi <- list(
  "P" = P,
  "I" = I,
  "C" = C,
  "Y" = t(citems),
  "mu_mean" = Mu_mean,
  "Mu_cov" = MU_cov,
  "lambda_mean" = Lambda_mean,
  "Lambda_cov" = LAMBDA_cov
)

# Standardized sum scores to initialize LV
sum_scores <- rowSums(citems)
theta_init <- (sum_scores - mean(sum_scores)) / sd(sum_scores)

# Fit the model to the data 
fit_2pclsi <- mdl_2pclsi$sample(
  data = stanls_2pclsi,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 0,
  iter_sampling = 5,
  # Important: We do not initialize lambda, because lambda can be negative in this model. Every category gets its own lambda, so if we set them all to by very positve that would not soak up in the right mode / in the right spot.
  init = function() list(theta = rnorm(P, mean = theta_init, sd = 0)),
  adapt_engaged = FALSE
)

# todo todo todo: Model does not run


modelCategoricalLogit_samples$draws(variables = "initLambda")

modelCategoricalLogit_samples = modelCategoricalLogit_stan$sample(
  data = modelOrderedLogit_data,
  seed = 121120222,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  #init = function() list(initLambda=rnorm(nItems*(maxCategory-1), mean=1, sd=.1))
  init = function() list(theta=rnorm(nObs, mean=thetaInit, sd=0))
)


# checking convergence
max(modelCategoricalLogit_samples$summary()$rhat, na.rm = TRUE)

print(modelCategoricalLogit_samples$summary(variables = c("mu", "lambda")), n=Inf)

11 mu[1,2]   1.08    1.07   0.288 0.290  0.613   1.55    1.00    1520.    2106.
 12 mu[2,2]   0.373   0.367  0.249 0.245 -0.0402  0.788   1.00    2144.    2668.
 13 mu[3,2]  -0.497  -0.495  0.271 0.266 -0.944  -0.0488  1.00    2134.    2668.
 14 mu[4,2]   0.559   0.559  0.278 0.287  0.111   1.03    1.00    1824.    2785.
 15 mu[5,2]   0.649   0.639  0.269 0.257  0.215   1.10    1.00    1657.    2300.
 16 mu[6,2]   0.639   0.637  0.282 0.277  0.178   1.12    1.00    1854.    2789.
 17 mu[7,2]  -0.310  -0.311  0.241 0.242 -0.698   0.0860  1.00    2237.    2585.
 18 mu[8,2]   0.338   0.335  0.269 0.280 -0.0939  0.786   1.00    1807.    2661.
 19 mu[9,2]  -0.339  -0.338  0.234 0.231 -0.728   0.0444  1.00    2129.    2745.
 20 mu[10,2] -1.46   -1.46   0.260 0.265 -1.90   -1.06    1.00    4217.    2979.
 21 mu[1,3]   0.949   0.943  0.305 0.305  0.455   1.46    1.00    1440.    2298.
 22 mu[2,3]  -0.734  -0.734  0.371 0.371 -1.35   -0.139   1.00    1535.    2439.
 23 mu[3,3]  -0.216  -0.217  0.260 0.265 -0.637   0.203   1.00    1803.    2711.
 24 mu[4,3]   0.146   0.154  0.337 0.334 -0.417   0.685   1.00    1617.    2786.
 25 mu[5,3]  -0.619  -0.601  0.412 0.406 -1.33    0.0341  1.00    1342.    2192.
 26 mu[6,3]  -0.0562 -0.0546 0.368 0.369 -0.685   0.544   1.00    1689.    2484.
 27 mu[7,3]  -1.16   -1.16   0.345 0.343 -1.75   -0.607   1.00    2688.    2463.
 28 mu[8,3]  -0.196  -0.192  0.350 0.350 -0.788   0.353   1.00    1420.    2507.
 29 mu[9,3]  -1.72   -1.70   0.410 0.410 -2.41   -1.08    1.00    1898.    2414.
 30 mu[10,3] -2.39   -2.37   0.393 0.395 -3.07   -1.78    1.00    3164.    2694.
 31 mu[1,4]   0.313   0.317  0.323 0.328 -0.225   0.836   1.00    1670.    2397.
 32 mu[2,4]  -1.13   -1.12   0.359 0.361 -1.73   -0.539   1.00    2686.    3013.
 33 mu[3,4]  -1.90   -1.88   0.409 0.404 -2.61   -1.26    1.00    3633.    2639.
 34 mu[4,4]  -1.31   -1.29   0.419 0.417 -2.02   -0.635   1.00    3379.    2848.
 35 mu[5,4]  -0.758  -0.749  0.378 0.382 -1.39   -0.146   1.00    2122.    2790.
 36 mu[6,4]  -1.11   -1.09   0.416 0.404 -1.81   -0.439   1.00    2767.    2661.
 37 mu[7,4]  -1.93   -1.91   0.392 0.380 -2.59   -1.31    1.00    3980.    3192.
 38 mu[8,4]  -1.76   -1.74   0.456 0.454 -2.54   -1.05    1.00    3348.    3064.
 39 mu[9,4]  -1.41   -1.40   0.328 0.322 -1.97   -0.885   1.00    3079.    2941.
 40 mu[10,4] -3.25   -3.22   0.480 0.487 -4.06   -2.50    1.00    4821.    2922.
 41 mu[1,5]  -0.975  -0.962  0.444 0.441 -1.71   -0.264   1.00    2396.    2969.
 42 mu[2,5]  -1.50   -1.48   0.398 0.394 -2.17   -0.872   1.00    3462.    2963.
 43 mu[3,5]  -1.97   -1.95   0.419 0.404 -2.70   -1.31    1.00    3781.    2957.
 44 mu[4,5]  -1.06   -1.05   0.386 0.385 -1.70   -0.434   1.00    3326.    3364.
 45 mu[5,5]  -1.39   -1.38   0.430 0.410 -2.13   -0.710   1.00    3043.    2806.
 46 mu[6,5]  -1.77   -1.75   0.488 0.484 -2.59   -0.999   1.00    3538.    2849.
 47 mu[7,5]  -2.29   -2.27   0.433 0.434 -3.03   -1.61    1.00    4175.    2684.
 48 mu[8,5]  -2.13   -2.11   0.503 0.509 -2.98   -1.34    1.00    4038.    3292.
 49 mu[9,5]  -1.91   -1.89   0.370 0.365 -2.52   -1.33    1.00    3361.    3098.
 50 mu[10,5] -2.21   -2.20   0.339 0.338 -2.79   -1.69    1.00    3877.    2339.
 61 lambda[…  1.20    1.19   0.236 0.228  0.827   1.61    1.00    1551.    2444.
 62 lambda[…  1.33    1.32   0.255 0.247  0.933   1.77    1.00    2392.    2556.
 63 lambda[…  1.59    1.58   0.289 0.288  1.12    2.08    1.00    2685.    2867.
 64 lambda[…  1.75    1.74   0.284 0.278  1.28    2.23    1.00    2389.    3093.
 65 lambda[…  1.60    1.59   0.283 0.279  1.15    2.09    1.00    2191.    2379.
 66 lambda[…  2.04    2.03   0.302 0.298  1.55    2.54    1.00    2429.    2694.
 67 lambda[…  1.44    1.42   0.269 0.261  1.02    1.91    1.00    2917.    2644.
 68 lambda[…  1.70    1.69   0.301 0.294  1.23    2.22    1.00    2471.    2824.
 69 lambda[…  1.37    1.36   0.256 0.253  0.972   1.81    1.00    2866.    2830.
 70 lambda[…  1.26    1.24   0.285 0.281  0.807   1.74    1.00    3078.    2793.
 71 lambda[…  1.92    1.91   0.281 0.284  1.47    2.39    1.00    1765.    2819.
 72 lambda[…  2.78    2.75   0.413 0.407  2.14    3.49    1.00    2449.    2732.
 73 lambda[…  1.79    1.79   0.294 0.297  1.33    2.30    1.00    2334.    2615.
 74 lambda[…  2.79    2.78   0.381 0.377  2.18    3.43    1.00    2266.    2891.
 75 lambda[…  3.35    3.33   0.451 0.442  2.63    4.12    1.00    2544.    2613.
 76 lambda[…  3.11    3.10   0.413 0.404  2.46    3.81    1.00    2481.    2598.
 77 lambda[…  2.28    2.26   0.374 0.366  1.69    2.93    1.00    2565.    2633.
 78 lambda[…  3.10    3.07   0.448 0.440  2.41    3.89    1.00    2411.    2789.
 79 lambda[…  2.55    2.54   0.403 0.409  1.89    3.22    1.00    2526.    2590.
 80 lambda[…  1.84    1.83   0.382 0.364  1.25    2.50    1.00    2856.    2961.
 81 lambda[…  1.46    1.46   0.289 0.289  1.00    1.94    1.00    1945.    2759.
 82 lambda[…  1.86    1.85   0.402 0.393  1.22    2.53    1.00    3276.    2749.
 83 lambda[…  1.59    1.58   0.414 0.412  0.914   2.29    1.00    3039.    2778.
 84 lambda[…  1.89    1.88   0.460 0.457  1.16    2.68    1.00    3797.    3113.
 85 lambda[…  2.27    2.25   0.425 0.426  1.59    2.99    1.00    3073.    3060.
 86 lambda[…  2.31    2.30   0.467 0.459  1.55    3.07    1.00    2731.    3069.
 87 lambda[…  1.63    1.63   0.425 0.427  0.932   2.35    1.00    3436.    3030.
 88 lambda[…  1.95    1.93   0.539 0.547  1.09    2.86    1.00    3080.    2322.
 89 lambda[…  1.56    1.54   0.361 0.354  0.986   2.16    1.00    3330.    2759.
 90 lambda[…  0.863   0.861  0.460 0.467  0.113   1.62    1.00    4392.    2698.
 91 lambda[…  2.07    2.06   0.429 0.423  1.39    2.81    1.00    2149.    2732.
 92 lambda[…  1.79    1.78   0.454 0.462  1.06    2.54    1.00    2544.    2430.
 93 lambda[…  1.77    1.76   0.438 0.446  1.06    2.50    1.00    2557.    2521.
 94 lambda[…  1.76    1.74   0.437 0.437  1.05    2.50    1.00    2873.    2917.
 95 lambda[…  2.11    2.09   0.521 0.525  1.26    2.99    1.00    2503.    2510.
 96 lambda[…  2.07    2.07   0.544 0.534  1.19    2.97    1.00    2917.    2837.
 97 lambda[…  1.56    1.54   0.455 0.455  0.830   2.33    1.00    2988.    2711.
 98 lambda[…  1.61    1.59   0.550 0.543  0.731   2.54    1.00    3735.    2802.
 99 lambda[…  1.49    1.47   0.421 0.413  0.800   2.21    1.00    2775.    2692.
100 lambda[…  1.27    1.25   0.363 0.367  0.697   1.87    1.00    2085.    2299.



## investigating option characteristic curves ===================================
itemNumber = 10

labelMu = paste0("mu[", itemNumber, ",", 1:5, "]")
muParams = modelCategoricalLogit_samples$summary(variables = labelMu)

labelLambda = paste0("lambda[", itemNumber, ",", 1:5, "]")
lambdaParams = modelCategoricalLogit_samples$summary(variables = labelLambda)

# item plot
theta = seq(-3,3,.1) # for plotting analysis lines--x axis values
thetaMat = NULL

logit=NULL
prob=NULL
probsum = 0
option = 1
for (option in 1:5){
  logit = cbind(logit, muParams$mean[option] + lambdaParams$mean[option]*theta)
  prob = cbind(prob, exp(logit[,option]))
  probsum = probsum+ exp(logit[,option])
}

for (option in 1:5){
  thetaMat = cbind(thetaMat, theta)
  prob[,option] = prob[,option]/probsum
}

matplot(x = thetaMat, y = prob, type="l", xlab=expression(theta), ylab="P(Y |theta)", 
        main=paste0("Option Characteristic Curves for Item ", itemNumber), lwd=3)

legend(x = -3, y = .8, legend = paste("Option", 1:5), lty = 1:5, col=1:5, lwd=3)


# Comparing EAP Estimates with Posterior SDs

plot(y = modelCategoricalLogit_samples$summary(variables = c("theta"))$sd, 
     x = modelCategoricalLogit_samples$summary(variables = c("theta"))$mean,
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)", main="Mean vs SD of Theta")

# Comparing EAP Estimates with Sum Scores
plot(y = rowSums(conspiracyItems), 
     x = modelCategoricalLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "Sum Score", xlab = expression(theta))

# Comparing Thetas: Categorical Logit vs Normal:
plot(y = modelCFA_samples$summary(variables = c("theta"))$mean, 
     x = modelCategoricalLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "Normal", xlab = "Categorical Logit")

# Comparing Theta SDs: Categorical Logit vs Normal:
plot(y = modelCFA_samples$summary(variables = c("theta"))$sd, 
     x = modelCategoricalLogit_samples$summary(variables = c("theta"))$sd,
     ylab = "Normal", xlab = "Categorical Logit", main="Posterior SDs")

# Which is bigger?
hist(modelCFA_samples$summary(variables = c("theta"))$sd-
       modelCategoricalLogit_samples$summary(variables = c("theta"))$sd,
     main = "SD(normal) - SD(categorical)")

# Comparing Thetas: Categorical Logit vs Ordinal:
plot(y = modelCategoricalLogit_samples$summary(variables = c("theta"))$mean, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "NRM", xlab = "GRM")

# Comparing Theta SDs: Ordered Logit vs Binomial:
plot(y = modelCategoricalLogit_samples$summary(variables = c("theta"))$sd, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     ylab = "NRM", xlab = "GRM", main="Posterior SDs")

# Which is bigger?
hist(modelCategoricalLogit_samples$summary(variables = c("theta"))$sd-
       modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     main = "SD(NRM) - SD(GRM)")
