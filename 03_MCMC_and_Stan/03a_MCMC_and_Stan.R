#################
# MCMC and Stan #
#################

library(cmdstanr)
# bayesplot: for plotting posterior distributions
library(bayesplot)
# HDInterval: for constructing Highest Density Posterior Intervals
library(HDInterval)
library(ggplot2)

########
# data #
########

# outcome: Weight in pounds
DietData <- read.csv(file = "DietData.csv")

# center predictor variable
# gives E(Y | X = 60) NOT 0 anymore
DietData$Height60IN <- DietData$HeightIN-60

#################
# visualization #
#################

# density weight in pounds 
ggplot(data = DietData, aes(x = WeightLB)) +
  geom_histogram(aes(y = ..density..), position = "identity", binwidth = 10) + 
  geom_density(alpha=.2)

# density weight in pounds  by group
ggplot(data = DietData, aes(x = WeightLB, color = factor(DietGroup), 
fill = factor(DietGroup))) + 
  geom_histogram(aes(y = ..density..), position = "identity", binwidth = 10) + 
  geom_density(alpha=.2) 

# regression lines: weightLBS = f(heightIN) varying by group
ggplot(data = DietData, aes(x = HeightIN, y = WeightLB, 
shape = factor(DietGroup), color = factor(DietGroup))) +
  geom_smooth(method = "lm", se = FALSE) +
  geom_point()

#################
# linear model  #
#################

# Asm  y = f(X) + e = beta0 + beta1*X + e
# where e ~ N(0, sigma_e^2)

###########################
# Ordinary least squares  #
###########################

# full analysis model suggested by data:
full_fml <- "WeightLB ~ Height60IN + factor(DietGroup) + Height60IN:factor(DietGroup)"
fullModel <- lm(formula = full_fml, data = DietData)

# examining assumptions and leverage of fit
plot(fullModel)

# looking at ANOVA table
anova(fullModel)

# looking at parameter summary
summary(fullModel)

# show empty model using OLS
empty_fml <- "WeightLB ~ 1"
emptyModel <- lm(formula = empty_fml, data = DietData)

# looking at ANOVA table
anova(emptyModel)

# looking at parameter summary
summary(emptyModel)

############
# bayesian #
############

# compile model -- this method is for stand-alone stan files
mdl00 <- cmdstan_model(stan_file = "./model00.stan", pedantic = TRUE)

# show location of executable
mdl00$exe_file()

# Stan syntax (also in model00.stan)
fml <- "

data {
  int<lower=0> N;
  vector[N] y;
}

parameters {
  real beta0;
  real<lower=0> sigma;
}

model {
  beta0 ~ normal(0, 1000); // prior for beta0
  sigma ~ uniform(0, 100000); // prior for sigma
  y ~ normal(beta0, sigma);
}

"

# compile model -- this method is for stan code as a string
mdl00 <- cmdstan_model(stan_file = write_stan_file(stanModel))

# show location of executable
mdl00$exe_file()

# build r list for stan
stanls <- list(
  N = nrow(DietData),
  y = DietData$WeightLB
)

# run MCMC chain (sample from posterior p.d.)
fit00 <- model00.fromFile$sample(
  data = stanData,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 10000,
  iter_sampling = 10000
)

# diagnositcs
#
# R-hat
fit00$cmdstan_diagnose()
fit00$print()
fit00$diagnostic_summary()

# plots
mcmc_trace(fit00$draws(c("beta0", "sigma")))
mcmc_dens(fit00$draws(c("beta0", "sigma")))

# examples of non-converged chains

mdl00_bad = mdl00$sample(
  data = stanData,
  seed = 112,
  chains = 4,
  parallel_chains = 4,
  # iter_warmup = 1e1, #
  iter_warmup = 1e3, # Good warm up, allow for fewer samples
  iter_sampling = 1e2 # No warum up, not much sample = bad results
# iter_sampling = 1e4 # We still get the right answer with a ton of samples
)

# time-series plot
mcmc_trace(mdl00_bad$draws(c("beta0", "sigma")))
# density plot
mcmc_dens(mdl00_bad$draws(c("beta0", "sigma")))

# summarize parameters (HPDI)
hdi(model00.samples$draws("beta0"))
hdi(model00.samples$draws("sigma"))
