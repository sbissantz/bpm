
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
conspiracy_items <- conspiracy_data[, 1 : 10]

# Number of items 
I <- ncol(conspiracy_items)

# NEVER DO THIS IN PRACTICE!
# Converting polytomous to dichotomous responses, loses data
# 0: strongly disagree or disagree;
# 1: neither, agree, and strongly disagree
items_bin <- conspiracy_items
for (var in seq_len(I)) {
  items_bin[which(items_bin[, var] <= 3), var] <- 0
  items_bin[which(items_bin[, var] > 3), var] <- 1
}

# Number of respondents
P <- nrow(items_bin)

# Examining data after transformation
table(items_bin$PolConsp1, items_bin$PolConsp1)
# Most peopls disagree with the first item

# Item means
# Note: the mean is the proportion of respondents who agreed with the item
colMeans(items_bin)

#################################
# Example: Likelihood Functions #
#################################

# Examine the data likelihood for the factor loading of the 1st item, lambda_1

# We have the jont p.m.f. f(Y_p| lambda_1, mu_1 theta_p)
# We fix mu_1 and theta_p
# ...for lambda1
mu_1 <- -2 # fix
theta <- rnorm(P, 0, 1) # fix as standardized LV
hist(theta)

# Assumption: observations are independent 
# f(Y_p| lambda_1, theta_p) = prod_{p=1}^{P} f(Y_p| lambda_1, theta_p)

# We let the loadings vary, to find the maximum (likelihood estimate)
lambda_1 <- seq(-2, 2, .01) # Loadings
log_lik <- vector("numeric", I)

# par <- 1 # for demonstrating
for (par in seq_along(lambda_1)) {
    # calculate the log-odd or logits
    logit <- mu_1 + lambda_1[par] * theta
    # Convert to probability
    p <- exp(logit) / (1 + exp(logit))
    # Plug the probability into the binomial p.m.f.
    # Define the log likelihood function
    # f(Y_p| lambda_1, theta_p) = prod_{p=1}^{P} f(Y_p| lambda_1, theta_p)
    # The product becomes a sum because of the log: log-likelihood
    # log f(Y_p| lambda_1, theta_p) = sum{p=1}^{P} f(Y_p| lambda_1, theta_p)
    LL_bern <- sum( # Sum over all persons
        dbinom( # Take the value of the binomial p.m.f.
            items_bin$PolConsp1, 1, p, # evaluate at all persons
            log = TRUE # on the log scale
        )
    )
    log_lik[par] <- LL_bern
}

# visualize
plot(x = lambda_1, y = log_lik, type = "l") # find the maximum  of LL
plot(x = lambda_1, y = -log_lik, type = "l") # find the maximum  of -LL

# examine the data likelihood for latent trait of the 2nd person, lambda_1

# We have the jont p.m.f. f(Y_p| lambda_1, mu_1 theta_p)
# We fix mu and lambda 
# .... for theta_2
mu <- runif(I, -2, 0) # fix
lambda <- runif(I, 0, 2) # fix
person <- 2 # fix

# We let the latent trait vary, to find the maximum (likelihood estimate)
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
plot(x = -theta, y = log_lik, type = "l") # No clear minimum!

################
# IRT: 2 PL SI #
################

# Compile model
mdl_2pl_si <- cmdstan_model("./stan/4c/2pl_si.stan", pedantic = TRUE)

# Item intercept hyperparameters
mu_mean <- rep(0, I)
Mu_cov <- diag(1000, I)

# item discrimination/factor loading hyperparameters
lambda_mean <- rep(0, I)
Lambda_cov <- diag(1000, I)

#############
# Stan list #
#############

# Build r list for stan
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

# Run MCMC chain (sample from posterior p.d.)
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

###############
# Diagnostics #
###############

# Assess convergence: summary of all parameters
fit_2pl_si$cmdstan_diagnose()
fit_2pl_si$diagnostic_summary()

# Checking convergence
max(fit_2pl_si$summary()$rhat, na.rm = TRUE)

###################
# Item parameters #
###################

# Summary of the item intercepts 
fit_2pl_si$summary(variables = "mu") # E(Y| theta = 0)

# Interpret item intercept 1 (remember: standardized LV): 
# "For someone with an average amount of the tendency to agree with conspiracy
# theories we would expect their log odds/logits for item one to be -2.46
fit_2pl_si$summary(variables = "mu")[1,]
# That is a ... percent chance to say "1" to the first item
plogis(-2.46) * 100

# Summary of the item discrimination/loadings 
fit_2pl_si$summary(variables = "lambda") # E(Y| theta + 1) - E(Y| theta)

# Interpret item discrimination/loading 1: 
# "If we compare two persons that differ by one (standard deviation) in their
# tendency to believe in conspiracy theories, we would expect the one with the
# higher tendency to have a 1.91 higher change to say "1" to item 1"
fit_2pl_si$summary(variables = "lambda")[1,] 
# That is a ... percent higher chance to say "1" to the first item
plogis(1.91) * 100

# Extract posterior draws
draws_si <- posterior::as_draws_rvars(fit_2pl_si$draws())

# Fixed theta values
theta_fixed <- seq(-3, 3, length.out = P)

# Drawing item characteristic curves for item
draws_si$logit <- draws_si$mu + draws_si$lambda * t(theta_fixed)
# ...including estimation uncertainty in theta
# draws$logit <- draws$mu + draws$lambda * t(draws$theta)

# Cannot use logistic function directly, because of the rvar data type
draws_si$y <- exp(draws_si$logit) / (1 + exp(draws_si$logit))

# Bernoulli ICC (item characteristic curve) for item 5
# i.e.: E(Y_5| theta) -- on the probability scale
itemno <- 5
plot(theta_fixed, mean(draws_si$y[itemno, ]), type = "l",
  main = paste("Item", itemno, "ICC"), ylim = c(0, 1), lwd = 2,
  xlab = expression(theta),
  ylab = paste("Item", itemno, "Retrodicted Value")
)
yno_arr <- posterior::draws_of(draws_si$y[itemno, ])
for (d in 1:100) {
  lines(theta_fixed, yno_arr[d, 1, ], col = "steelblue", lwd = 0.5)
}
lines(theta_fixed, mean(draws_si$y[itemno, ]), lwd = 5)
legend(-3, 1,
  legend = c("Posterior Draw", "EAP"),
  col = c("steelblue", "black"), lty = c(1, 1), lwd = 5
)

# Investigating item parameters

# Item intercepts
mcmc_trace(fit_2pl_si$draws(variables = "mu"))
mcmc_dens(fit_2pl_si$draws(variables = "mu"))
# Results are pretty skewed

# Loadings
mcmc_trace(fit_2pl_si$draws(variables = "lambda"))
mcmc_dens(fit_2pl_si$draws(variables = "lambda"))
# Results are pretty skewed

# Bivariate posterior p.d.
mcmc_pairs(fit_2pl_si$draws(), pars = c("mu[1]", "lambda[1]"))
# Even though we specified the prior p.d.s for the parameter independently, the
# posterior p.d.s are not independent.
# Which makes sense -- we use lines; thus slope and intercepts are not 
# independent If the slope is high, the intercept is low and vice versa.

# investigating the latent variables
fit_2pl_si$summary(variables = "theta")

# EAP Estimates of Latent Variables
hist(mean(draws_si$theta),
  main = "EAP Estimates of Theta",
  xlab = expression(theta)
)
theta_mean_arr <- draws_of(rvar_mean(draws_si$theta))
for (i in 1:20) {
  # Estimation uncertainty in the EAP estimates of theta
  abline(v = theta_mean_arr[i], col = "steelblue", lwd = 2)
}
# Mean of the EAP estimates of theta (mean of posterior means)
abline(v = mean(theta_mean_arr), lwd = 5)
# The average tendency to believe in conspiracies across persons is estimated 
# to be very low -- centered at 0, but quite skewed.

# Comparing two posterior p.d.s of theta
plot(c(-3, 3), c(0, 2), type = "n", xlab = expression(theta), ylab = "Density")
lines(density(draws_of(draws_si$theta[1])), col = "red", lwd = 3)
lines(density(draws_of(draws_si$theta[2])), col = "blue", lwd = 3)
# Note the difference in the spread of the two posterior p.d.s
# Theta 1 does not provide additional info beyond the prior, p.d., because
# person 1 did not answer any items with a "1".
person <- 1 ; items_bin[person, ]

# Comparing EAP Estimates with Posterior SDs
# Posterior SD is a clearly defined function of theta!
# in MLE this is akin to the conditional S.E.M; each theta has a SE and the 
# sizes vary depending on the value of theta!
plot(mean(draws_si$theta), sd(draws_si$theta),
  pch = 19,
  xlab = "E(theta|Y)", ylab = "SD(theta|Y)",
  main = "Mean vs SD of Theta"
)

# Comparing EAP Estimates with Sum Scores
plot(mean(draws_si$theta), rowSums(items_bin),
  pch = 19,
  ylab = "Sum Score", xlab = expression(theta)
)
# The EAP estimate is not necessarily the sum score (non-linear trend).
# In the Rasch model the sum score is still a sufficient statistic for theta

##################
# IRT: 2PL SI II #
##################
# Discrimination/difficulty calculated in the generated quantities block

# compile model
mdl_2pl_si2 <- cmdstan_model("./stan/4c/2pl_si2.stan", pedantic = TRUE)

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


# Diagnostics 
max(fit_2pl_si2$summary()$rhat, na.rm = TRUE)
fit_2pl_si2$cmdstan_diagnose()
fit_2pl_si2$diagnostic_summary()

# item parameter results
print(fit_2pl_si2$summary(variables = c("mu", "lambda")), n = Inf) # SI
print(fit_2pl_si2$summary(variables = c("a", "b")), n = Inf) # DD

# extract posterior draws
draws_si2 <- posterior::as_draws_rvars(fit_2pl_si2$draws())

###############
# IRT: 2PL DD #
###############
# Slope/intercept calculated in the generated quantities block

# Compile model
mdl_2pl_dd <- cmdstan_model("./stan/4c/2pl_dd.stan", pedantic = TRUE)

# Item intercept hyperparameters
b_mean <- rep(0, I)
b_var_hp <- 1000
B_cov = diag(b_var_hp, I)

# Item discrimination/factor loading hyperparameters
a_mean <- rep(0, I)
a_var_hp <- 1000
A_cov <- diag(a_var_hp, I)

# Stan list
stanls_2pl_dd <- list(
  "P" = P,
  "I" = I,
  "Y" = t(items_bin), 
  "b_mean" = b_mean,
  "B_cov" = B_cov,
  "a_mean" = a_mean,
  "A_cov" = A_cov
)

# Fit model to data
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
# Diagnostics #
###############

# Checking convergence
fit_2pl_dd$cmdstan_diagnose()
fit_2pl_dd$diagnostic_summary()
max(fit_2pl_dd$summary()$rhat, na.rm = TRUE)

###################
# Item parameters #
###################

# Summary of the discrimination/loading parameter
fit_2pl_dd$summary("a")

# Interpret the discrimination/loading parameter alpha_1:
# The item has moderate capability to distinguish between individuals with
# lower and higher levels of conspiracy belief.
fit_2pl_dd$summary("a")[1, ]

# Summary of the difficulty parameter
fit_2pl_dd$summary("b")

# Interpret the difficulty parameter beta_1:
# A respondent with a belief in conspiracies of 1.48 (on the theta scale: SD) 
# has a 50% probability of agreeing with item 1
fit_2pl_dd$summary("b")[1, ]

# Extract posterior draws
draws_dd <- posterior::as_draws_rvars(fit_2pl_dd$draws())

# Fixed theta values
theta_fixed <- seq(-3, 3, length.out = P)

# Drawing item characteristic curves for item
draws_dd$logit <- draws_dd$mu + draws_dd$lambda * t(theta_fixed)
# ...including estimation uncertainty in theta
# draws_dd$y <- exp(draws_dd$logit) / (1 + exp(draws_dd$logit))

##############
# Comparison #
##############

# Comparing b EAP estimates
cor(mean(draws_dd$b), mean(draws_si2$b))

# Comparing with parameters estimates from 
# Note: The jitter in the values comes from sampling variation
plot(mean(draws_dd$b), mean(draws_si2$b),
  xlab = "Discrimination/Difficulty Model",
  ylab = "Slope/Intercept Model",
  main = "Difficulty Parameter EAP Estimates",
  pch = 19, cex = 2, xlim = c(1, 2), ylim = c(1, 2)
)
abline(a = 0, b = 1, lty = 2, lwd = 2)

# Comparing with other parameters estimated:
# Note: The jitter in the values comes from sampling variation
plot(mean(draws_dd$theta), mean(draws_si2$theta),
  xlab = "Discrimination/Difficulty Model",
  ylab = "Slope/Intercept Model",
  main = "Difficulty Parameter EAP Estimates",
  pch = 19, cex = 2, xlim = c(1, 2), ylim = c(1, 2)
)
abline(a = 0, b = 1, lty = 2, lwd = 2)

# Comparing difficulty EAP estimates
cor(mean(draws_dd$theta), mean(draws_si2$theta))
cor(sd(draws_dd$theta), sd(draws_si2$theta))

# Comparing with other parameters estimated:
plot(sd(draws_dd$theta), sd(draws_si2$theta),
  xlab = "Discrimination/Difficulty Model",
  ylab = "Slope/Intercept Model",
  main = "Theta SD Estimates",
  pch = 19, cex = 2
)

########################
# Auxiliary Statistic  #
########################

# Compile model
mdl_2pl_dd2 <- cmdstan_model("./stan/4c/2pl_dd2.stan", pedantic = TRUE)

# Item intercept hyperparameters
b_mean <- rep(0, I)
b_var_hp <- 1000
B_cov = diag(b_var_hp, I)

# Item discrimination/factor loading hyperparameters
a_mean <- rep(0, I)
a_var_hp <- 1000
A_cov <- diag(a_var_hp, I)

# Values for auxiliary statistics
theta_fix <- seq(-3, 3, length.out = P)

# Stan list
stanls_2pl_dd2 <- list(
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

# Fit model to the data
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
# Diagnostics #
###############

# checking cnvergence
fit_2pl_dd2$summary()
fit_2pl_dd2$cmdstan_diagnose()
fit_2pl_dd2$diagnostic_summary()
max(fit_2pl_dd2$summary()$rhat, na.rm = TRUE)

###################
# Item parameters #
###################

# summary of the item parameters
fit_2pl_dd2$summary("a") # E(Y| theta = 0)

# Interpret the discrimination/loading parameter alpha_1:
# The item has moderate capability to distinguish between individuals with
# lower and higher levels of conspiracy belief.
fit_2pl_dd2$summary("a")[1, ]

fit_2pl_dd2$summary("b") # E(Y| theta + 1) - E(Y| theta)

# Interpret the difficulty parameter beta_1:
# A respondent with a belief in conspiracies of 1.48 (on the theta scale: SD) 
# has a 50% probability of agreeing with item 1
# Note: The person is above average (average: 1 – standardized LV)
fit_2pl_dd2$summary("b")[1, ]

# Extract posterior draws
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
itemno <- 2
item_info_arr <- draws_of(draws_dd2$item_info[, itemno])
item_info_max <- max(item_info_arr)
plot(theta_fix, mean(draws_dd2$item_info[, itemno]),
     xlab = expression(theta), 
     ylab = "Information", type = "l",
     main = paste0(itemno, " Information Function"), lwd = 2,
     ylim = c(-1, item_info_max))
# Include uncertainty
for (d in seq_len(100)) {
  lines(x = theta_fix, y = item_info_arr[d, , ], col = "steelblue")
}
# EAP TCC
lines(theta_fix, mean(draws_dd2$item_info[, itemno]), lwd = 7)

# Visualize: Test Information Function
tif_arr <- draws_of(draws_dd2$test_info)
tif_max <- 2000 
plot(theta_fix, mean(draws_dd2$test_info),
  xlab = expression(theta),
  ylab = "Information", type = "l",
  main = "Test Information Function", lwd = 2,
  ylim = c(0, tif_max)
)
for (d in seq_len(100)) {
  lines(theta_fix, tif_arr[d, ], col = "steelblue")
}
# EAP TIF
lines(theta_fix, mean(draws_dd2$test_info), lwd = 7)

# EAP TCC
tcc_arr <- draws_of(draws_dd2$test_info)
tcc_max <- 2000 
plot(theta_fix, mean(draws_dd2$test_info),
     ylim = c(-1, tcc_max),
     xlab = expression(theta), 
     ylab = "Information", type = "l",
     main = "Test Information Function", 
     lwd = 2)
for (d in seq_len(100)) {
  lines(theta_fix, tif_arr[d, ], col = "steelblue")
}
# EAP TIF
lines(theta_fix, mean(draws_dd2$test_info), lwd = 7)

####################
# Other IRT Models #
####################

#############################
# IRT: 1PL DD (Rasch model) #
#############################

# Compile model
mdl_1pl_dd <- cmdstan_model("./stan/4c/1pl_dd.stan", pedantic = TRUE)

# Item intercept hyperparameters
b_mean <- rep(0, I)
b_var_hp <- 1000
B_cov = diag(b_var_hp, I)

# Stan list
stanls_1pl_dd <- list(
  "P" = P,
  "I" = I,
  "Y" = t(items_bin),
  "b_mean" = b_mean,
  "B_cov" = B_cov 
)

# Fit the model to the data
fit_1pl_dd <- mdl_1pl_dd$sample(
  data = stanls_1pl_dd,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000
)

###############
# Diagnostics #
###############

fit_1pl_dd$summary()
fit_1pl_dd$cmdstan_diagnose()
fit_1pl_dd$diagnostic_summary()
max(fit_1pl_dd$summary()$rhat, na.rm = TRUE)

#####################
# Person parameters #
#####################

# Interpret the difficulty parameter beta_1:
# A respondent with a belief in conspiracies of 1.99 (on the theta scale: SD) 
# has a 50% probability of agreeing with item 1
# Note: The person is above average (average: 1 – standardized LV)
fit_1pl_dd$summary("b")[1, ]

################################
# IRT: 3PL DD (Birnbaum model) #
################################

# Compile the model
mdl_3pl_dd <- cmdstan_model("./stan/4c/3pl_dd.stan", pedantic = TRUE)

# Stan list
stanls_3pl_dd <- list(
  "P" = P,
  "I" = I,
  "Y" = t(items_bin),
  "b_mean" = b_mean,
  "B_cov" = B_cov,
  "a_mean" = a_mean,
  "A_cov" = A_cov
)

# Fit the model to the data
fit_3pl_dd <- mdl_3pl_dd$sample(
  data = stanls_3pl_dd,
  seed = 021120222,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  init = function() list("a" = rnorm(I, mean = 5, sd = 1))
)

###############
# Diagnostics #
###############

fit_3pl_dd$summary()
fit_3pl_dd$cmdstan_diagnose()
fit_3pl_dd$diagnostic_summary()
max(fit_3pl_dd$summary()$rhat, na.rm = TRUE)

###########
# Summary #
###########

print(fit_3PL$summary(variables = c("a", "b", "c")), n = Inf)

#####################################
# IRT: 2PNO (Normal Ogive Model) DD #
#####################################

# Compile model
mdl_2pno_dd <- cmdstan_model("./stan/4c/2pno_dd.stan", pedantic = TRUE)

# Item intercept hyperparameters
b_mean <- rep(0, I)
b_var_hp <- 1000
B_cov = diag(b_var_hp, I)

# Item discrimination/factor loading hyperparameters
a_mean <- rep(0, I)
a_var_hp <- 1000
A_cov <- diag(a_var_hp, I)

# Stan list
stanls_2pno_dd <- list(
  "P" = P,
  "I" = I,
  "Y" = t(items_bin),
  "b_mean" = b_mean,
  "B_cov" = B_cov,
  "a_mean" = a_mean,
  "A_cov" = A_cov
)

# Fit the model to the data
fit_2pno_dd <- mdl_2pno_dd$sample(
  data = stanls_2pno_dd,
  seed = 021120221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  init = function() list("a" = rnorm(I, mean = 3, sd = .05))
)

###############
# Diagnostics #
###############

# checking cnvergence
fit_2pno_dd$summary()
fit_2pno_dd$cmdstan_diagnose()
fit_2pno_dd$diagnostic_summary()
max(fit_2pno_dd$summary()$rhat, na.rm = TRUE)

###################
# Item parameters #
###################

# summary of the item parameters
fit_2pno_dd$summary("a") # E(Y| theta = 0)

# Interpret the discrimination/loading parameter alpha_1:
# The item has moderate capability to distinguish between individuals with
# lower and higher levels of conspiracy belief.
fit_2pno_dd$summary("a")[1, ]

fit_2pno_dd$summary("b") # E(Y| theta + 1) - E(Y| theta)

# Interpret the difficulty parameter beta_1:
# A respondent with a belief in conspiracies of 1.07 (on the theta scale: SD) 
# has a 50% probability of agreeing with item 1
# Note: The person is above average (average: 1 – standardized LV)
fit_2pno_dd$summary("b")[1, ]
