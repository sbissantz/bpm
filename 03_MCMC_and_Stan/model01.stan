data {
  int<lower=0> N; // Sample size
  vector[N] weightLB; // Outcome variable
  vector[N] height60IN; // Centered height variable
  vector[N] group2; // Dummy variable for group 2
  vector[N] group3; // Dummy variable for group 3
  vector[N] heightXgroup2; // Interaction height times group 2
  vector[N] heightXgroup3; // Interaction height times group 3
}

parameters {
  real beta0; // Intercept 
  real betaHeight; // Conditional slope for height
  real betaGroup2; // Conditional main effect of group2
  real betaGroup3; // Conditional main effect of group3
  real betaHxG2; // Height by group2 interaction
  real betaHxG3; // Height by group3 interaction
  real<lower=0> sigma; // Residual standard deviation
}

model {
  weightLB ~ normal(
    beta0 + betaHeight * height60IN + betaGroup2 * group2 + 
    betaGroup3 * group3 + betaHxG2 *heightXgroup2 +
    betaHxG3 * heightXgroup3, sigma);
}