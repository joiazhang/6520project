# STSCI 6520 Project
STSCI 6520: Statistical Computing, Fall 2023, Minjie Jia and Joia Zhang

To install our package, in R first load library(devtools). Then you can use the command install_github("joiazhang/6520project", subdir="onlinegrad") to install the package, and finally library(onlinegrad) to load the package.

Our package has three functions my_OGD, my_adagrad, and my_adam for online gradient descent, adaptive gradient descent, and adaptive moment estimation, respectively. These functions are meant to be used for linear regression or logistic regression. To see the documentation for the functions about input and output values, you can run ?my_OGD for example.

To see an example of data generated for linear and logistic regression, and also examples of the functions being used and plotting commands to visualize the results, see the file project.Rmd.
