data1<-read.csv('traffic_900.csv');
library(forecast);
train<-data1[1:600,1];
test<-data1[601:900,1];
#train<-matrix(train);
#test<-matrix(test);

# off line training
sensor<-ts(train,frequency=15);
fit <- auto.arima(sensor,approximation=FALSE,trace=FALSE);

#----------------------------------------------------------
# fit results show that it is a (1,0,0)(2,0,0)15 sarima model
# mathetical equation is (1-FAI1*B^15 - FAI2*B^30)(1-fai1*B)*Xt = zt
# remove seasonality sar1 means lag is 15, sar2 means lag is 30
# ------season-----------------------
train_season <- 0.116* train[1:570] + 0.411 * train[16:585]
data1_season <- 0.116* data1[1:870,1] + 0.411 * data1[16:885,1]
write.csv(data1_season, 'seasonal_before_kf.csv')

#---------arma----------------------------
train_no_season <- train[31:600] - train_season 
plot(train_no_season)
data1_no_season <- data1[31:900,1] - 0.116 * data1[1:870,1] - 0.411 * data1[16:885,1]
train_no_season_arma <- 0.754*train_no_season[1:569] # 2:570 arma part + 30 -> 32:600
data1_no_season_arma <- 0.754 * data1_no_season[1:869] # 2:870 arma + 30 -> 32:900 part
write.csv(data1_no_season_arma,'kf_arima.csv')

#SARIMA model (1,0,0)(2,0,0)15 gives us a seasonal part, a AR(1) part and a residual part
#the ar(1) part is estimated using a state space and kalman filter
#the residual part is now also estimated using garch and then also a state space and kalman filter
#----residuals for garch(1,1)-------------------
train_residuals <- train_no_season[2:570] - train_no_season_arma 
plot(train_residuals)
data1_no_season_residuals <- data1_no_season[2:870] - data1_no_season_arma 
write.csv(data1_no_season_residuals^2,'kf_garch.csv')

#----------------new approach------------------------------------
# timeserie_traffic = ts(data1[1:900,1])
# library(forecast)
# trend_traffic = ma(timeserie_traffic, order = 15, centre = T)
# plot(as.ts(timeserie_traffic))
# lines(trend_traffic)
# plot(as.ts(trend_traffic))
# 
# # trend dont need be updated one by one, because from the equation, it's always the same
# 
# for (i in c(601:900)){
#   ts_tem <- ts(data1[1:i,1], frequency=15);
#   
# }

#----------------DLM part-----------------------------------------

# simulate a AR(1) model using state space function
# Yt = F*xt + V
# xt = G*xt-1 + W
require(dlm)
tsbuildkf1 = function(parm) {
  mod = dlmModPoly(order = 1, dV = exp(parm[1]), dW = c(exp(parm[2])))
  mod$FF[1] = parm[3] # for a vector, we have to add [1] 
  mod$GG[1] = parm[4]
  return(mod)
}

y = ts(train_no_season_arma)
#fit the model on the training set using maximum likelihood
# rep(0,2) for V and W, rep(1,2) means initialize 1 for FF and GG
fitkf1 = dlmMLE(y, c(rep(0,2), rep(1,2)), tsbuildkf1)
#return four parameters
#R-V-exp(5.36) - 212.72
#Q-W-exp(13.47) - 707858.9
#H-FF-0.0571
#A-GG-0.9127


tsbuildkf2 = function(parm) {
  mod = dlmModPoly(order = 3, dV = exp(parm[1]), dW = c(exp(parm[2]), exp(parm[3]), exp(parm[4])))
  mod$FF[1] = parm[5] # for a vector, we have to add [1] 
  mod$FF[2] = parm[6]
  mod$FF[3] = parm[7]
  mod$GG[1,1] = parm[8]
  mod$GG[2,2] = parm[9]
  mod$GG[3,3] = parm[10]
  return(mod)
}
y= ts(train_residuals^2) # calculate square
# FF is a 1 by 3 matrix, GG is a 3 by 3 matrix
fitkf2 = dlmMLE(y, c(rep(0,3), rep(1,7)), tsbuildkf2)


#R-V-exp(0.683)  - 1.979
#Q-W-
#[exp(0.006)             [1.01
# exp(0.1008)         -->   1.10
# exp(11.390)]             88432.96]
# H-FF
#[30.57, -34.61, 0.45]
# A-GG
#[0.16
#       0.994
#             -0.106]


#--------------calculate the prediction and PIs------------
traffic_season <- read.csv('seasonal_before_kf.csv')
traffic_arma <- read.csv('kf_arima_predicted_N_100.csv')
traffic_variance <- read.csv('kf_garch_predicted_N_200.csv')
# the last 300 are the predictions by kalman filter
traffic_season <- traffic_season[(nrow(traffic_season) - 300 +1):nrow(traffic_season), 2]
traffic_arma <- traffic_arma[!is.na(traffic_arma[,2]),2]
traffic_arma <- traffic_arma[(length(traffic_arma) - 300 +1):length(traffic_arma)]
traffic_variance <- traffic_variance[(nrow(traffic_variance) - 300 + 1):nrow(traffic_variance),2]
traffic_sd <- sqrt(traffic_variance)

traffic_pre <- traffic_season + traffic_arma
traffic_pre90_lo <- traffic_pre - 1.64 * traffic_sd
traffic_pre90_hi <- traffic_pre + 1.64 * traffic_sd

traffic_pre95_lo <- traffic_pre - 1.96 * traffic_sd
traffic_pre95_hi <- traffic_pre + 1.96 * traffic_sd

traffic_pre99_lo <- traffic_pre - 2.58 * traffic_sd
traffic_pre99_hi <- traffic_pre + 2.58 * traffic_sd

res <- cbind(traffic_pre, traffic_pre90_lo, traffic_pre90_hi, traffic_pre95_lo, traffic_pre95_hi, 
             traffic_pre99_lo, traffic_pre99_hi)
write.csv(res, "Guo_2014.csv")
