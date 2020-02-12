

rm(list=ls())


working_dir <- c('/Dropbox/loki_0/simple_rt_experiment_probabilityC/analysis/circ_glm/')


library(dplyr)
library(broom)
library(tidyverse)
library(circglmbayes)
library(reshape2)
library(boot)


if (Sys.info()['sysname'] == 'Darwin'){
  home = '/Users/67981492/'
} else if (Sys.info()['sysname'] == 'Linux'){
  home = '/data/'
}

setwd(paste0(home, working_dir))


load(paste0(home, working_dir, 'angular_decay.RData'))


# establish what parameters _should_ be

Q = 10000
burnin = 2000
thin = 1

total_its = ((Q*thin) + burnin)
saved_its = Q


# check mcmc parameters 

print(paste('checking mcmc params ...'))

stopifnot(angular_decay_m$thin == thin)
stopifnot(angular_decay_m$burnin == burnin)
stopifnot(angular_decay_m$TotalIts == total_its)
stopifnot(angular_decay_m$SavedIts == saved_its)



# plot traces to assess convergence 


print(paste('plotting traces for subject ...'))

print(plot(angular_decay_m, type='tracestack'))


 
# get autocorrelation of traces for all coefficients 

# more complex models have high autocorrelation... 


print(paste('plotting acf ...'))

print(acf(angular_decay_m$kp_chain)) 
print(acf(angular_decay_m$b0_chain)) 
print(acf(angular_decay_m$mu_chain)) 

  

# get dist. of traces for all coefficients 

for (col in seq(ncol(angular_decay_m$all_chains))){
  
  print(hist(angular_decay_m$all_chains[, col])) 
  
}


# check if the point estimate is outside the 95% credible interval of the posterior distribution for all coefficients 

stopifnot((angular_decay_m$kp_mean > angular_decay_m$kp_HDI[1]) & (angular_decay_m$kp_mean < angular_decay_m$kp_HDI[2]))  # concentration param.
stopifnot((angular_decay_m$b0_meandir > angular_decay_m$b0_CCI[1]) & (angular_decay_m$b0_meandir < angular_decay_m$b0_CCI[2]))  # intercept

if (!all(is_empty(angular_decay_m$bt_CCI))){
  for (col in seq(ncol(angular_decay_m$bt_mean))){
    
    stopifnot((angular_decay_m$bt_mean[col] > angular_decay_m$bt_CCI[1, col]) & (angular_decay_m$bt_mean[col] < angular_decay_m$bt_CCI[2, col])) # check if mean for slope est are outside the credible intervals
  }}

if (!all(is_empty(angular_decay_m$dt_CCI))){
  for (col in seq(ncol(angular_decay_m$dt_meandir))){
    
    stopifnot((angular_decay_m$dt_meandir[col] > angular_decay_m$dt_CCI[1, col]) & (angular_decay_m$dt_meandir[col] < angular_decay_m$dt_CCI[2, col])) # check if mean for the difference estimates are cred. int.
  
  }} 


# implement the bic calcs? need mult models. estimate a null. 
      
  

