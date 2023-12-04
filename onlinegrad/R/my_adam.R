#' Adam for regression
#' 
#' Adaptive moment estimation for linear regression.
#' @param X An n x p matrix of predictors where rows are observations and columns are predictors.
#' @param Y An n x 1 vector quantitative response variable.
#' @param lr A constant that is the learning rate.
#' @param beta_0 An p x 1 vector that is the initialization for the coefficients.
#' @param rho_1 constant for 1st moment decay rate
#' @param rho_2 constant for 2nd moment decay rate
#' @param epsilon positive constant for nonzero/invertibility
#' @param regression Boolean, regression if true, else classification.
#' @return List where first elemnt is an n x p matrix where each ith row is the coefficients for the ith iteration and the columns are predictors and second element is a nx1 vector of runtimes for each iteration.
#' @examples
#' my_adam(X=X, Y=Y, lr=0.0000001, beta_0=rep(0, p), rho_1=0.9, rho_2=0.999, epsilon=1e-8)
#' 
#' @export
my_adam = function(X, Y, lr, beta_0, rho_1, rho_2, epsilon, regression=T) {
  n = nrow(X)
  p = ncol(X)
  betahats = matrix(nrow=n, ncol=p)
  Ms = matrix(nrow=n, ncol=p) # 1st moment estimate
  Rs = matrix(nrow=n, ncol=p) # 2nd moment estimate
  Mhats = matrix(nrow=n, ncol=p) # 1st moment bias correction
  Rhats = matrix(nrow=n, ncol=p) # 2nd moment bias correction
  runtimes = rep(NA, n)
  runtimes[1] = 0
  
  # initialize
  betahats[1, ] = beta_0
  Ms[1, ] = rep(0, p)
  Rs[1, ] = rep(0, p)
  
  if (regression) {
    for (t in 1:(n-1)) {
      start_time = Sys.time()
      
      x_t = as.matrix(X[t, ])
      beta_t = as.matrix(betahats[t, ])
      y_t_hat = t(beta_t)%*%x_t
      Y_t = Y[t]
      d_loss = 2*beta_t%*%t(x_t)%*%x_t - 2*x_t%*%Y_t
      Ms[t+1, ] = rho_1*as.matrix(Ms[t, ]) + (1-rho_1)*d_loss
      Rs[t+1, ] = rho_2*as.matrix(Rs[t, ]) + (1-rho_2)*d_loss^2
      Mhats[t+1, ] = Ms[t, ] / (1-rho_1^t)
      Rhats[t+1, ] = Rs[t, ] / (1-rho_2^t)
      
      betahats[t+1, ] = beta_t - lr*(Mhats[t+1, ]/(sqrt(Rhats[t+1, ]+epsilon))) # update
      
      end_time = Sys.time()
      runtimes[t+1] = runtimes[t] + (end_time - start_time)
    } # end for
  } else {
    # classification
    
    for (t in 1:(n-1)) {
      start_time = Sys.time()
      
      x_t = as.matrix(X[t, ])
      beta_t = as.matrix(betahats[t, ])
      y_t_hat = t(beta_t)%*%x_t
      Y_t = Y[t]
      
      # predict y
      Z = t(beta_t)%*%x_t # omit intercept bs
      Y_pred = 1/(1+1/exp(Z))
      
      d_loss = (1/n)*as.numeric(Y_pred-Y_t)*x_t
      
      Ms[t+1, ] = rho_1*as.matrix(Ms[t, ]) + (1-rho_1)*d_loss
      Rs[t+1, ] = rho_2*as.matrix(Rs[t, ]) + (1-rho_2)*d_loss^2
      Mhats[t+1, ] = Ms[t, ] / (1-rho_1^t)
      Rhats[t+1, ] = Rs[t, ] / (1-rho_2^t)
      
      betahats[t+1, ] = beta_t - lr*(Mhats[t+1, ]/(sqrt(Rhats[t+1, ]+epsilon))) # update
      
      end_time = Sys.time()
      runtimes[t+1] = runtimes[t] + (end_time - start_time)
    } # end for
  } # end else
  return(list(betahats, runtimes))
}

