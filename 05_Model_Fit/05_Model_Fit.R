
#########
# Setup #
#########

library(cmdstanr)
  # set number of cores to 4 for this analysis
  options(mc.cores = 4)
library(bayesplot)
library(ggplot2)
library(posterior)

set.seed(112)

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

#######################################################
# 2 POL SI (ordered-logit) aka. Graded Response Model #
# (Slope-Intercept Form)                              #
# ...with posterior predictive model checking         #
#######################################################

# Compile model
mdl_2polsi <- cmdstan_model("./stan/5/2pol_si_ppc.stan", pedantic = TRUE)

# Item threshold hyperparameters
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
print(fit_2polsi$summary(c("lambda", "mu")), n = Inf)

#########
# Draws #
#########

draws_2polsi <- posterior::as_draws_rvars(fit_2polsi$draws())

# Extract only the simulated data
Y_sim <- posterior::as_draws_matrix(fit_2polsi$draws("Y_sim"))

# Reshape Y_sim into a 3D array of dimensions 8000 x 10 x 177
Yarr_sim <- array(Y_sim, dim = c(8000, 10, 177))
# Reshape the array into a 177 x 10 x 8000 array 
# ... some structure I can imagine (a cube)
Yt_sim <- aperm(Yarr_sim, c(3, 2, 1))

# Calculate the mean of the simulated data
PPC <- list()
PPC$mean <- t(apply(Yt_sim, c(2,3), mean))

# Plot the density of the mean of the first item
plot(density(PPC$mean[, 1]), main = "Posterior predictive p.d.: Item 1 Mean")
# Add a vertical line for the observed mean
abline(v = mean(citems$PolConsp1), lty = 2, col = 2, lwd = 3)

# Get all unique combinations
n_combi <- choose(I, 2)
combi <- combn(1:I, 2) 

# Select the appropriate item
PPC$cor <- matrix(NA, nrow = S, ncol = n_combi)
cor_obs <- vector(length = n_combi)
for (pair in seq_len(n_combi)) {
  for (s in seq_len(S)) {
    PPC$cor[s, pair] <- cor(
      Yt_sim[, combi[1, pair], s],
      Yt_sim[, combi[2, pair], s]
    )
  }
  cor_obs[pair] <- cor(citems[,combi[1, pair]], citems[,combi[2, pair]])
}
combi_nms <- paste0(paste0("i", combi[1, ]), "_", paste0("i", combi[2, ]))
colnames(PPC$cor) <- combi_nms
names(cor_obs) <- combi_nms

# Get correlations of item 1 and 2
plot(density(PPC$cor[,1]), main = "Item 1 Item 2 Pearson Correlation")
# Add a vertical line for the observed correlation
abline(v = cor_obs[1], lty = 2, col = 2, lwd = 3)

# Average PP correlation
PPC$cor_mean <- colMeans(PPC$cor)

# Correlation summary
cor_summary <- data.frame(
  "cor_obs" = cor_obs,
  "ppc_mcor" = PPC$cor_mean,
  "residual" = cor_obs - PPC$cor_mean,
  "cor_obs_pct" = NA,
  "tail" = NA
)

# Add some additional information
for (pair in seq_len(n_combi)) {
  ecdf_cor <- ecdf(PPC$cor[, pair])
  pct <- ecdf_cor(cor_obs[pair])
  cor_summary$cor_obs_pct[pair] <- pct
  tail <- ifelse(pct > .975 | pct < 0.025, TRUE, FALSE)
  cor_summary$tail[pair] <- tail
}
cor_summary # Better: Polychoric correlations!

# Interpet: 98.5 % of the simulated correlations are bigger than oberserved.
cor_summary["i1_i10", "cor_obs_pct"] # 0.01575
#...tail value

baditms_nms <- strsplit(combi_nms[cor_summary$tail], "_")
# How many times does each item show up in a misfitting correlation?
baditms_nms |> unlist() |> table() 

# TODO - Multidimensional Model
