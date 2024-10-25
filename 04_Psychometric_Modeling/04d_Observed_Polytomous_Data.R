
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
fit_binom2pl_si <- mdl_binom2pl_si$sample(
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

#########
# Draws #
#########

# Extract posterior draws
draws_binom2pl_si <- posterior::as_draws_rvars(fit_binom2pl_si$draws())

# Fixed theta values
theta_fixed <- seq(-3, 3, length.out = P)

draws_binom2pl_si$logodds <- with(
  draws_binom2pl_si, t(mu + lambda * t(theta_fixed)) # t() again to make P X I
)
# Include estimation uncertainty in theta
#draws_binom2pl_si$logodds <- with(
  #draws_binom2pl_si, t(mu + lambda * t(theta))
#)

draws_binom2pl_si$prob <- with(
  # Convert to probabilit (logit not implemented in rvars)
  draws_binom2pl_si, exp(logodds) / (1 + exp(logodds))
)

# Visualize OCCs
# Option characteristic curves
itemno <- 10
mean(draws_binom2pl_si$prob)
plot(c(-3, 3), c(0, 1), type = "n", main = "Option Characteristic Curve"
      xlab = expression(theta), ylab = "P(Y |theta)")
draws_binom2pl_si$prob[, itemno]
prob10_arr <- draws_of(draws_binom2pl_si$prob[, itemno])
for (o in seq_len(O)) { # For each of the five options
  for (d in 1:100) {    # For each draw
    # lines(theta_fixed, dbinom(1, T, prob = prob10_arr[d, , ]))
    lines(theta_fixed, dbinom(o - 1, T, prob = prob10_arr[d, , ]),
          col = o, lwd = 0.3
    )
  }
  # EAP lineas
  lines(theta_fixed, dbinom(o - 1, T,
        prob = mean(draws_binom2pl_si$prob[, itemno])), lwd = 4)
}
legend("topright", legend = paste("Option", 1:5), col = 1:5, lwd = 3)

# todo todo todo

# investigating item parameters ================================================
itemNumber = 10

labelMu = paste0("mu[", itemNumber, "]")
labelLambda = paste0("lambda[", itemNumber, "]")
itemParameters = modelBinomial_samples$draws(variables = c(labelMu, labelLambda), format = "draws_matrix")
itemSummary = modelBinomial_samples$summary(variables = c(labelMu, labelLambda))

# item plot
theta = seq(-3,3,.1) # for plotting analysis lines--x axis values

# drawing item characteristic curves for item
y = 4*exp(as.numeric(itemParameters[1,labelMu]) + as.numeric(itemParameters[1,labelLambda])*theta)/
  (1+exp(as.numeric(itemParameters[1,labelMu]) + as.numeric(itemParameters[1,labelLambda])*theta)) +1
plot(x = theta, y = y, type = "l", main = paste("Item", itemNumber, "ICC"), 
     ylim=c(0,6), xlab = expression(theta), ylab=paste("Item", itemNumber,"Expected Value"))
for (draw in 2:nrow(itemParameters)){
  y =4*exp(as.numeric(itemParameters[draw,labelMu]) + as.numeric(itemParameters[draw,labelLambda])*theta)/
    (1+exp(as.numeric(itemParameters[draw,labelMu]) + as.numeric(itemParameters[draw,labelLambda])*theta)) +1 
  lines(x = theta, y = y)
}

# drawing limits
lines(x = c(-3,3), y = c(5,5), type = "l", col = 4, lwd=5, lty=2)
lines(x = c(-3,3), y = c(1,1), type = "l", col = 4, lwd=5, lty=2)

# drawing EAP line
y = 4*exp(itemSummary$mean[which(itemSummary$variable==labelMu)] + 
  itemSummary$mean[which(itemSummary$variable==labelLambda)]*theta)/
  (1+exp(itemSummary$mean[which(itemSummary$variable==labelMu)] + 
           itemSummary$mean[which(itemSummary$variable==labelLambda)]*theta)) +1
lines(x = theta, y = y, lwd = 5, lty=3, col=2)

# legend
legend(x = -3, y = 4, legend = c("Posterior Draw", "Item Limits", "EAP"), col = c(1,4,2), lty = c(1,2,3), lwd=5)


# EAP Estimates of Latent Variables
hist(modelBinomial_samples$summary(variables = c("theta"))$mean, main="EAP Estimates of Theta", 
     xlab = expression(theta))

# Comparing Two Posterior Distributions
theta1 = "theta[1]"
theta2 = "theta[2]"
thetaSamples = modelBinomial_samples$draws(variables = c(theta1, theta2), format = "draws_matrix")
thetaVec = rbind(thetaSamples[,1], thetaSamples[,2])
thetaDF = data.frame(observation = c(rep(theta1,nrow(thetaSamples)), rep(theta2, nrow(thetaSamples))), 
                     sample = thetaVec)
names(thetaDF) = c("observation", "sample")
ggplot(thetaDF, aes(x=sample, fill=observation)) +geom_density(alpha=.25)

# Comparing EAP Estimates with Posterior SDs
plot(y = modelBinomial_samples$summary(variables = c("theta"))$sd, 
     x = modelBinomial_samples$summary(variables = c("theta"))$mean,
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)", main="Mean vs SD of Theta")

# Comparing EAP Estimates with Sum Scores
plot(y = rowSums(conspiracyItemsBinomial), 
     x = modelBinomial_samples$summary(variables = c("theta"))$mean,
     ylab = "Sum Score", xlab = expression(theta))

# Comparing Thetas: Binomial vs Normal:
plot(y = modelCFA_samples$summary(variables = c("theta"))$mean, 
     x = modelBinomial_samples$summary(variables = c("theta"))$mean,
     ylab = "Normal", xlab = "Binomial")

# Ordered Logit (Multinomial/categorical distribution) Model Syntax =======================



modelOrderedLogit_syntax = "

data {
  int<lower=0> nObs;                            // number of observations
  int<lower=0> nItems;                          // number of items
  int<lower=0> maxCategory; 
  
  array[nItems, nObs] int<lower=1, upper=5>  Y; // item responses in an array

  array[nItems] vector[maxCategory-1] meanThr;   // prior mean vector for intercept parameters
  array[nItems] matrix[maxCategory-1, maxCategory-1] covThr;      // prior covariance matrix for intercept parameters
  
  vector[nItems] meanLambda;         // prior mean vector for discrimination parameters
  matrix[nItems, nItems] covLambda;  // prior covariance matrix for discrimination parameters
}

parameters {
  vector[nObs] theta;                // the latent variables (one for each person)
  array[nItems] ordered[maxCategory-1] thr; // the item thresholds (one for each item category minus one)
  vector[nItems] lambda;             // the factor loadings/item discriminations (one for each item)
}

model {
  
  lambda ~ multi_normal(meanLambda, covLambda); // Prior for item discrimination/factor loadings
  theta ~ normal(0, 1);                         // Prior for latent variable (with mean/sd specified)
  
  for (item in 1:nItems){
    thr[item] ~ multi_normal(meanThr[item], covThr[item]);             // Prior for item thresholds
    Y[item] ~ ordered_logistic(lambda[item]*theta, thr[item]);
  }
  
}

generated quantities{
  array[nItems] vector[maxCategory-1] mu;
  for (item in 1:nItems){
    mu[item] = -1*thr[item];
  }
}

"

modelOrderedLogit_stan = cmdstan_model(stan_file = write_stan_file(modelOrderedLogit_syntax))


# Data needs: successive integers from 1 to highest number (recode if not consistent)
maxCategory = 5

# data dimensions
nObs = nrow(conspiracyItems)
nItems = ncol(conspiracyItems)

# item threshold hyperparameters
thrMeanHyperParameter = 0
thrMeanVecHP = rep(thrMeanHyperParameter, maxCategory-1)
thrMeanMatrix = NULL
for (item in 1:nItems){
  thrMeanMatrix = rbind(thrMeanMatrix, thrMeanVecHP)
}

thrVarianceHyperParameter = 1000
thrCovarianceMatrixHP = diag(x = thrVarianceHyperParameter, nrow = maxCategory-1)
thrCovArray = array(data = 0, dim = c(nItems, maxCategory-1, maxCategory-1))
for (item in 1:nItems){
  thrCovArray[item, , ] = diag(x = thrVarianceHyperParameter, nrow = maxCategory-1)
}

# item discrimination/factor loading hyperparameters
lambdaMeanHyperParameter = 0
lambdaMeanVecHP = rep(lambdaMeanHyperParameter, nItems)

lambdaVarianceHyperParameter = 1000
lambdaCovarianceMatrixHP = diag(x = lambdaVarianceHyperParameter, nrow = nItems)


modelOrderedLogit_data = list(
  nObs = nObs,
  nItems = nItems,
  maxCategory = maxCategory,
  maxItem = maxItem,
  Y = t(conspiracyItems), 
  meanThr = thrMeanMatrix,
  covThr = thrCovArray,
  meanLambda = lambdaMeanVecHP,
  covLambda = lambdaCovarianceMatrixHP
)

modelOrderedLogit_samples = modelOrderedLogit_stan$sample(
  data = modelOrderedLogit_data,
  seed = 121120221,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 5000,
  init = function() list(lambda=rnorm(nItems, mean=5, sd=1))
)

# checking convergence
max(modelOrderedLogit_samples$summary()$rhat, na.rm = TRUE)

# item parameter results
print(modelOrderedLogit_samples$summary(variables = c("lambda", "mu")) ,n=Inf)


## investigating option characteristic curves ===================================
itemNumber = 10

labelMu = paste0("mu[", itemNumber, ",", 1:4, "]")
labelLambda = paste0("lambda[", itemNumber, "]")
muParams = modelOrderedLogit_samples$summary(variables = labelMu)
lambdaParams = modelOrderedLogit_samples$summary(variables = labelLambda)

# item plot
theta = seq(-3,3,.1) # for plotting analysis lines--x axis values
y = NULL
thetaMat = NULL
expectedValue = 0

option = 1
for (option in 1:5){
  if (option==1){
    prob = 1 - exp(muParams$mean[which(muParams$variable == labelMu[option])] + 
                 lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta)/
      (1+exp(muParams$mean[which(muParams$variable == labelMu[option])] + 
               lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta))
  } else if (option == 5){
    
    prob = (exp(muParams$mean[which(muParams$variable == labelMu[option-1])] + 
                  lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta)/
              (1+exp(muParams$mean[which(muParams$variable == labelMu[option-1])] + 
                       lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta)))
  } else {
    prob = (exp(muParams$mean[which(muParams$variable == labelMu[option-1])] + 
                  lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta)/
              (1+exp(muParams$mean[which(muParams$variable == labelMu[option-1])] + 
                       lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta))) -
      exp(muParams$mean[which(muParams$variable == labelMu[option])] + 
            lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta)/
      (1+exp(muParams$mean[which(muParams$variable == labelMu[option])] + 
               lambdaParams$mean[which(lambdaParams$variable == labelLambda[1])]*theta))
  }
  
  thetaMat = cbind(thetaMat, theta)
  expectedValue = expectedValue + prob*option
  y = cbind(y, prob)
}

matplot(x = thetaMat, y = y, type="l", xlab=expression(theta), ylab="P(Y |theta)", 
        main=paste0("Option Characteristic Curves for Item ", itemNumber), lwd=3)

legend(x = -3, y = .8, legend = paste("Option", 1:5), lty = 1:5, col=1:5, lwd=3)

## plot of EAP of expected value per item ======================================================
plot(x = theta, y = expectedValue, type = "l", main = paste("Item", itemNumber, "ICC"), 
     ylim=c(0,6), xlab = expression(theta), ylab=paste("Item", itemNumber,"Expected Value"), lwd = 5, lty=3, col=2)

# drawing limits
lines(x = c(-3,3), y = c(5,5), type = "l", col = 4, lwd=5, lty=2)
lines(x = c(-3,3), y = c(1,1), type = "l", col = 4, lwd=5, lty=2)

# EAP Estimates of Latent Variables
hist(modelOrderedLogit_samples$summary(variables = c("theta"))$mean, 
     main="EAP Estimates of Theta", 
     xlab = expression(theta))

# Comparing Two Posterior Distributions
theta1 = "theta[1]"
theta2 = "theta[2]"
thetaSamples = modelOrderedLogit_samples$draws(
  variables = c(theta1, theta2), format = "draws_matrix")
thetaVec = rbind(thetaSamples[,1], thetaSamples[,2])
thetaDF = data.frame(
  observation = c(rep(theta1,nrow(thetaSamples)), rep(theta2, nrow(thetaSamples))), 
  sample = thetaVec)
names(thetaDF) = c("observation", "sample")
ggplot(thetaDF, aes(x=sample, fill=observation)) +geom_density(alpha=.25)

# Comparing EAP Estimates with Posterior SDs

plot(y = modelOrderedLogit_samples$summary(variables = c("theta"))$sd, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$mean,
     xlab = "E(theta|Y)", ylab = "SD(theta|Y)", main="Mean vs SD of Theta")

# Comparing EAP Estimates with Sum Scores
plot(y = rowSums(conspiracyItems), 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "Sum Score", xlab = expression(theta))

# Comparing Thetas: Ordered Logit vs Normal:
plot(y = modelCFA_samples$summary(variables = c("theta"))$mean, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "Normal", xlab = "Ordered Logit")

# Comparing Theta SDs: Ordered Logit vs Normal:
plot(y = modelCFA_samples$summary(variables = c("theta"))$sd, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     ylab = "Normal", xlab = "Ordered Logit", main="Posterior SDs")

# Which is bigger?
hist(modelCFA_samples$summary(variables = c("theta"))$sd-
       modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     main = "SD(normal) - SD(ordered)")

# Comparing Thetas: Ordered Logit vs Binomial:
plot(y = modelBinomial_samples$summary(variables = c("theta"))$mean, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$mean,
     ylab = "Binomial", xlab = "Ordered Logit")

# Comparing Theta SDs: Ordered Logit vs Binomial:
plot(y = modelBinomial_samples$summary(variables = c("theta"))$sd, 
     x = modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     ylab = "Binomial", xlab = "Ordered Logit", main="Posterior SDs")

# Which is bigger?
hist(modelBinomial_samples$summary(variables = c("theta"))$sd-
       modelOrderedLogit_samples$summary(variables = c("theta"))$sd,
     main = "SD(binomial) - SD(ordered)")



# Categorical Logit (Multinomial/categorical distribution) Model Syntax =======================
# Also known as: Nominal Response Model (in IRT literature) 


modelCategoricalLogit_syntax = "
data {
  int maxCategory;
  int nObs;
  int nItems;
  
  array[nItems, nObs] int Y; 
  
  array[nItems] vector[maxCategory-1] meanMu;   // prior mean vector for intercept parameters
  array[nItems] matrix[maxCategory-1, maxCategory-1] covMu;      // prior covariance matrix for intercept parameters
  
  array[nItems] vector[maxCategory-1] meanLambda;       // prior mean vector for discrimination parameters
  array[nItems] matrix[maxCategory-1, maxCategory-1] covLambda;  // prior covariance matrix for discrimination parameters
  
}

parameters {
  array[nItems] vector[maxCategory - 1] initMu;
  array[nItems] vector[maxCategory - 1] initLambda;
  vector[nObs] theta;                // the latent variables (one for each person)
}

transformed parameters {
  array[nItems] vector[maxCategory] mu;
  array[nItems] vector[maxCategory] lambda;
  
  for (item in 1:nItems){
    mu[item, 2:maxCategory] = initMu[item, 1:(maxCategory-1)];
    mu[item, 1] = 0.0; // setting one category's intercept to zero
    
    lambda[item, 2:maxCategory] = initLambda[item, 1:(maxCategory-1)];
    lambda[item, 1] = 0.0; // setting one category's lambda to zero
    
  }
}

model {
  
  vector[maxCategory] probVec;
  
  theta ~ normal(0,1);
  
  for (item in 1:nItems){
    for (category in 1:(maxCategory-1)){
      initMu[item, category] ~ normal(meanMu[item, category], covMu[item, category, category]);  // Prior for item intercepts
      initLambda[item, category] ~ normal(meanLambda[item, category], covLambda[item, category, category]);  // Prior for item loadings
    }
  }
    
  for (obs in 1:nObs) {
    for (item in 1:nItems){
      for (category in 1:maxCategory){
        probVec[category] = mu[item, category] + lambda[item, category]*theta[obs];     
      }
      Y[item, obs] ~ categorical_logit(probVec);
    }  
  }
}
"


modelCategoricalLogit_stan = cmdstan_model(stan_file = write_stan_file(modelCategoricalLogit_syntax))

# Data needs: successive integers from 1 to highest number (recode if not consistent)
maxCategory = 5

# data dimensions
nObs = nrow(conspiracyItems)
nItems = ncol(conspiracyItems)

# item threshold hyperparameters
muMeanHyperParameter = 0
muMeanVecHP = rep(muMeanHyperParameter, maxCategory-1)
muMeanMatrix = NULL
for (item in 1:nItems){
  muMeanMatrix = rbind(muMeanMatrix, muMeanVecHP)
}

muVarianceHyperParameter = 1
muCovarianceMatrixHP = diag(x = muVarianceHyperParameter, nrow = maxCategory-1)
muCovArray = array(data = 0, dim = c(nItems, maxCategory-1, maxCategory-1))
for (item in 1:nItems){
  muCovArray[item, , ] = diag(x = muVarianceHyperParameter, nrow = maxCategory-1)
}

# item discrimination/factor loading hyperparameters
lambdaMeanHyperParameter = 0
lambdaMeanVecHP = rep(lambdaMeanHyperParameter, maxCategory-1)
lambdaMeanMatrix = NULL
for (item in 1:nItems){
  lambdaMeanMatrix = rbind(lambdaMeanMatrix, lambdaMeanVecHP)
}

lambdaVarianceHyperParameter = 1
lambdaCovarianceMatrixHP = diag(x = lambdaVarianceHyperParameter, nrow = maxCategory-1)
lambdaCovArray = array(data = 0, dim = c(nItems, maxCategory-1, maxCategory-1))
for (item in 1:nItems){
  lambdaCovArray[item, , ] = diag(x = lambdaVarianceHyperParameter, nrow = maxCategory-1)
}


modelOrderedLogit_data = list(
  nObs = nObs,
  nItems = nItems,
  maxCategory = maxCategory,
  maxItem = maxItem,
  Y = t(conspiracyItems), 
  meanMu = muMeanMatrix,
  covMu = muCovArray,
  meanLambda = lambdaMeanMatrix,
  covLambda = lambdaCovArray
)

# for checking initial values:
modelCategoricalLogit_samples = modelCategoricalLogit_stan$sample(
  data = modelOrderedLogit_data,
  seed = 121120222,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 0,
  iter_sampling = 5,
  init = function() list(initLambda=rnorm(nItems*(maxCategory-1), mean=-1, sd=.1)), 
  adapt_engaged = FALSE
)
modelCategoricalLogit_samples$draws(variables = "initLambda")

modelCategoricalLogit_samples = modelCategoricalLogit_stan$sample(
  data = modelOrderedLogit_data,
  seed = 121120222,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  init = function() list(initLambda=rnorm(nItems*(maxCategory-1), mean=1, sd=.1))
)

# checking convergence
max(modelCategoricalLogit_samples$summary()$rhat, na.rm = TRUE)

print(modelCategoricalLogit_samples$summary(variables = c("mu", "lambda")), n=Inf)

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
