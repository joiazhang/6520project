#' Adagrad (diagonal) for regression
#' 
#' Adaptive gradient descent for linear regression.
#' @param X An n x p matrix of predictors where rows are observations and columns are predictors.
#' @param Y An n x 1 vector quantitative response variable.
#' @param lr A constant that is the learning rate.
#' @param beta_0 An p x 1 vector that is the initialization for the coefficients.
#' @param regression Boolean, regression if true, else classification.
#' @return List where first elemnt is an n x p matrix where each ith row is the coefficients for the ith iteration and the columns are predictors and second element is a nx1 vector of runtimes for each iteration.
#' 
#' @export
my_adagrad = function(X, Y, lr=1e-4, beta_0=rep(0, ncol(X)), regression=T) {
  # initialize variables
  n = nrow(X)
  p = ncol(X)
  betahats = matrix(nrow=n, ncol=p)
  betahats[1, ] = beta_0
  g_vec = matrix(nrow=n, ncol=p) # save matrix for the gradients where each gradient g_t is the t^{th} row of the matrix
  G_t = matrix(data=rep(0, p^2), nrow=p, ncol=p) # matrix that is a cumulative sum
  runtimes = rep(NA, n)
  runtimes[1] = 0
  
  if (regression) {
    for (t in 1:(n-1)) {
      start_time = Sys.time()
      
      # define values for update step
      x_t = as.matrix(X[t, ])
      beta_t = as.matrix(betahats[t, ])
      y_t_hat = t(beta_t)%*%x_t
      Y_t = Y[t]
      g_vec[t, ] = 2*beta_t%*%t(x_t)%*%x_t - 2*x_t%*%Y_t
      g_t = as.matrix(g_vec[t, ])
      G_t = G_t + g_t%*%t(g_t)
      diag_G_t = diag(diag(G_t), nrow=p, ncol=p)
      
      betahats[t+1, ] = beta_t - lr*as.matrix(diag(diag(diag_G_t^(-1/2)), nrow=p, ncol=p))%*%g_t # update step (diagonal)
      
      end_time = Sys.time()
      runtimes[t+1] = runtimes[t] + (end_time - start_time)
    } # end for
  } else {
    # classification
    
    for (t in 1:(n-1)) {
      start_time = Sys.time()
      
      # define values for update step
      x_t = as.matrix(X[t, ])
      beta_t = as.matrix(betahats[t, ])
      y_t_hat = t(beta_t)%*%x_t
      Y_t = Y[t]
      Z = t(beta_t)%*%x_t # sigmoid function (omitted intercept bs)
      Y_pred = 1/(1+1/exp(Z)) # predict y
      g_vec[t, ] = (1/n)*as.numeric(Y_pred-Y_t)*x_t
      g_t = as.matrix(g_vec[t, ])
      G_t = G_t + g_t%*%t(g_t)
      diag_G_t = diag(diag(G_t), nrow=p, ncol=p)
      
      betahats[t+1, ] = beta_t - lr*as.matrix(diag(diag(diag_G_t^(-1/2)), nrow=p, ncol=p))%*%g_t # update step (diaogonal)
      
      end_time = Sys.time()
      runtimes[t+1] = runtimes[t] + (end_time - start_time)
    }
    
  }
  return(list(betahats, runtimes))
}




