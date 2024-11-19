
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

# Compile model
mdl_2pclsi <- cmdstan_model("./stan/4d/2pcl_si.stan", pedantic = TRUE)

# Hyperparameters

# Item threshold hyperparameters
mu_mean <- rep(0, I)
Mu_mean <- replicate(C - 1, mu_mean) 

MU_cov <- replicate(C - 1, rep(0, I)) # 10 x 4
MU_cov <- array(0, dim = c(10, 4, 4)) # 10 x 4 x 4
for(d in seq_len(I)) {
  MU_cov[d , ,] <- diag(1, C - 1)
}

# Discrimination hyperparameters
lambda_mean <- rep(0, I)
Lambda_mean <- replicate(C - 1, mu_mean) 

LAMBDA_cov <- replicate(C - 1, rep(0, I)) # 10 x 4
LAMBDA_cov <- array(0, dim = c(10, 4, 4)) # 10 x 4 x 4
for(d in seq_len(I)) {
  LAMBDA_cov[d , ,] <- diag(1, C - 1)
}

# Standardized sum scores to initialize LV
sum_scores <- rowSums(citems)
theta_init <- (sum_scores - mean(sum_scores)) / sd(sum_scores)

# Stan list
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

# Check initial values
fit_2pclsi <- mdl_2pclsi$sample(
  data = stanls_2pclsi,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 0,
  iter_sampling = 5,
  init = function() list("theta" = rnorm(P, mean = theta_init, sd = 0)),
  adapt_engaged = FALSE
)
fit_2pclsi$draws(variables = "lambda_init")

# todo todo : What is adapt_engaged?

fit_2pclsi <- mdl_2pclsi$sample(
  data = stanls_2pclsi,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  #init = function() list(initLambda=rnorm(nItems*(maxCategory-1), mean=1, sd=.1))
  # Important: We do not initialize lambda, because lambda can be negative in this model. Every category gets its own lambda, so if we set them all to by very positve that would not soak up in the right mode / in the right spot.
  init = function() list("theta" = rnorm(P, mean = theta_init, sd = 0))
)

# todo todo

fit_2pclsi$summary(variables = "mu_init")


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
