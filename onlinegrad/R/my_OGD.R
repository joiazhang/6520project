#' OGD for regression
#' 
#' Online gradient descent for linear regression.
#' @param X An n x p matrix of predictors where rows are observations and columns are predictors.
#' @param Y An n x 1 vector quantitative response variable.
#' @param lr A constant that is the learning rate.
#' @param beta_0 An p x 1 vector that is the initialization for the coefficients.
#' @param regression Boolean, regression if true, else classification
#' @return List where first elemnt is an n x p matrix where each ith row is the coefficients for the ith iteration and the columns are predictors and second element is a nx1 vector of runtimes for each iteration.
#' @examples
#' my_OGD(X=X, Y=Y, lr=0.00001, beta_0=rep(0, ncol(X)))
#' my_OGD(X=X, Y=Y, lr=0.00001, beta_0=runif(ncol(X)))
#' 
#' @export
my_OGD = function(X, Y, lr, beta_0, regression=T) {
  n = nrow(X)
  p = ncol(X)
  betahats = matrix(nrow=n, ncol=p)
  betahats[1, ] = beta_0
  runtimes = rep(NA, n)
  runtimes[1] = 0
  if (regression) {
    for (t in 1:(n-1)) {
      start_time = Sys.time()
      x_t = as.matrix(X[t, ])
      beta_t = as.matrix(betahats[t, ])
      y_t_hat = t(beta_t)%*%x_t
      Y_t = Y[t]
      d_loss = 2*beta_t%*%t(x_t)%*%x_t - 2*x_t%*%Y_t
      betahats[t+1, ] = beta_t - lr*d_loss
      end_time = Sys.time()
      runtimes[t+1] = runtimes[t] + (end_time - start_time)
    } # end for
  } else {
    # classification
    bs = rep(NA, n) # intercept b for n iterations
    bs[1] = 0
    for (t in 1:(n-1)) {
      start_time = Sys.time()
      
      x_t = as.matrix(X[t, ])
      beta_t = as.matrix(betahats[t, ])
      Y_t = Y[t]
      
      # predict y
      Z = t(beta_t)%*%x_t+bs[t]
      Y_pred = 1/(1+1/exp(Z))
      
      # calculate gradient
      d_w = (1/n)*as.numeric(Y_pred-Y_t)*x_t
      d_b = (1/n)*sum(Y_pred-Y_t)
      
      # update coefficients and intercept
      betahats[t+1, ] = beta_t - lr*d_w
      bs[t+1] = bs[t] - lr*d_b
      
      end_time = Sys.time()
      runtimes[t+1] = runtimes[t] + (end_time - start_time)
    }
  } # end else
  return(list(betahats, runtimes, as.matrix(bs)))
}



