
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

# Number of categories
C <- max(citems)

# Number of dimensions
D <- 2
# Test if the data are conformable with two factors: 
# (1) A government factor & (2) a non-goverment factor 

############
# Q-Matrix #
############

# Initialize a I X D matrix with zeros
Q <- matrix(0, nrow = I, ncol = D)
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

#########################
# Multidimensional 2POL #
# MD-GRM                #   
#########################

# Compile model
mdl_md2polsi <- cmdstan_model("./stan/4e/md2pol_si.stan", pedantic = TRUE)

# Item threshold hyperparameter
Thr_mean <- replicate(C - 1, rep(0, I)) # 10 x 4
THR_cov <- array(0, dim = c(10, 4, 4)) # 10 x 4 x 4
for(d in seq_len(I)) {
  THR_cov[d , ,] <- diag(10, C - 1)
}

# Item discrimination/factor loading hyperparameters
lambda_mean <- rep(0, I)
Lambda_cov <- diag(10, I)

# Latent trait hyperparameters
Theta_mean <- rep(0, 2)

#############
# Stan list #
#############

stanls_md2polsi <- list(
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
  "theta_mean" = Theta_mean
)

# Initialization values
lambda_init <- rnorm(I, mean = 5, sd = 1)
sum_scores <- as.matrix(citems) %*% Q
theta_init <- scale(sum_scores)

# Run MCMC chain (sample from posterior p.d.)
fit_md2polsi <- mdl_md2polsi$sample(
  data = stanls_md2polsi,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000,
  iter_sampling = 2000,
  # Mean should be below 10, since the log of it is too large
  init = function() list("lambda" = lambda_init, "theta" = theta_init)
)

###############
# Diagnostics #
###############

# Checking convergence
fit_md2polsi$cmdstan_diagnose()
fit_md2polsi$diagnostic_summary()
max(fit_md2polsi$summary()$rhat, na.rm = TRUE)
# Important: If possible, reparameterize the model

# EES & overall
print(fit_md2polsi$summary(variables = c("lambda", "mu", "Theta_cor")), n = Inf)

#########
# Draws #
#########
draws_md2polsi <- posterior::as_draws_rvars(fit_md2polsi$draws())

# Posterior distribution of latent correlation
mcmc_trace(fit_md2polsi$draws(variables = "Theta_cor[1,2]"))
mcmc_dens(fit_md2polsi$draws(variables = "Theta_cor[1,2]"))