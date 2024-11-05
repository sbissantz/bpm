data {
  int P;                // Number of persons
  int I;                // Number of items
  int C;                // Number of categories
  array[I, P] int Y     // Item response array
}

parameters {
  // lambda
  // theta
  array[I] ordered[C-1] thr;      // Array of ordered intercepts

}
model {
  for(i in 1:I) {
    // Vectorized over persons
    Y ~ ordered_logistic(lambda[i] * theta, thr[i]);
  }
}
generated quantities {
  array[I] ordered[C-1] mu;      // Array of ordered intercepts
  for(i in 1:I) {
    mu = -1*thr[i];              // From tresholds to intercepts
  }
}