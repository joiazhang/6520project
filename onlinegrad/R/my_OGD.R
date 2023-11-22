#' OGD for regression
#' 
#' Online gradient descent for linear regression.
#' @param X An n x p matrix of predictors where rows are observations and columns are predictors.
#' @param Y An n x 1 vector quantitative response variable.
#' @param lr A constant that is the learning rate.
#' @param beta_0 An p x 1 vector that is the initialization for the coefficients.
#' @return An n x p matrix where each ith row is the coefficients for the ith iteration and the columns are predictors.
#' @examples
#' my_OGD(X=X, Y=Y, lr=0.00001, beta_0=rep(0, ncol(X)))
#' my_OGD(X=X, Y=Y, lr=0.00001, beta_0=runif(ncol(X)))
#' 
#' @export
my_OGD = function(X, Y, lr, beta_0) {
  n = nrow(X)
  p = ncol(X)
  betahats = matrix(nrow=n, ncol=p)
  betahats[1, ] = beta_0
  for (t in 1:(n-1)) {
    x_t = as.matrix(X[t, ])
    beta_t = as.matrix(betahats[t, ])
    y_t_hat = t(beta_t)%*%x_t
    Y_t = Y[t]
    # betahats[t+1, ] = beta_t - lr*as.numeric(y_t_hat-Y_t)*x_t # least means squares
    d_loss = 2*beta_t%*%t(x_t)%*%x_t - 2*x_t%*%Y_t
    betahats[t+1, ] = beta_t - lr*d_loss
  }
  return(betahats)
}



