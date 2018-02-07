library(TSA)
library(forecast)
library(rugarch)

dt<-read.csv("traffic_900.csv",na.strings=c('NA',''),stringsAsFactors=F)

res <- data.frame()
colnames(res) <- c("prediction", "low90", "high90", "low95", "high95", "low99", "high99")

i <- 600
while (i< 900){
  y <- dt[1:i,]

  pe <- periodogram(y, plot = FALSE)
  # abline(h=0)
  index <- order(pe$spec, decreasing = TRUE)
  value <- sort(pe$spec)
  # 600
  #----
  # frequency 40, 6, 80, 4, 120, 3 
  #= 15, 100, 7.5, 150, 5, 200 hour
  # first 6 largest value
  f1 <- index[1]/i
  f2 <- index[2]/i
  f3 <- index[3]/i
  f4 <- index[4]/i
  f5 <- index[5]/i
  f6 <- index[6]/i
  
  t<-c(1:(i+1))
  cosf1 <- cos(2*pi*f1*t)
  sinf1 <- sin(2*pi*f1*t)
  cosf2 <- cos(2*pi*f2*t)
  sinf2 <- sin(2*pi*f2*t)
  cosf3 <- cos(2*pi*f3*t)
  sinf3 <- sin(2*pi*f3*t)
  cosf4 <- cos(2*pi*f4*t)
  sinf4 <- sin(2*pi*f4*t)
  cosf5 <- cos(2*pi*f5*t)
  sinf5 <- sin(2*pi*f5*t)
  cosf6 <- cos(2*pi*f6*t)
  sinf6 <- sin(2*pi*f6*t)

  
  df<-data.frame(cosf1,sinf1,cosf2,sinf2,cosf3,sinf3,cosf4,sinf4,cosf5, sinf5,cosf6,sinf6)
  
  # use first i to train, 
  lr1 <- lm(y~., data = df[1:i,])
  #http://stats.stackexchange.com/questions/69144/calculating-prediction-interval
  # predict i+1
  lr1_90 <- predict(lr1, df[i+1,], interval="predict", level = 0.90) 
  lr1_95 <- predict(lr1, df[i+1,], interval="predict", level = 0.95)
  lr1_99 <- predict(lr1, df[i+1,], interval="predict", level = 0.99)
  
  #plot(lr1$residuals, type = 'l')
  #plot(lr1$fitted.values, type = 'l')
  
  # arima model 
  # provide the mean part of the traffic volume
  arima1 <- auto.arima(lr1$residuals)
  #plot(arima1$residuals, type = 'l')
  
  # ---------------acf test----------------------
  # plot acf -comment off
  # acf(resid(arima1)) 
  # Ljung-Box Statistics for ARIMA residuals, p value less than 0.05, means the residuals is not independent
  Box.test(resid(arima1),type="Ljung", lag=15, fitdf = 1)
  
  ar_pre <- forecast(arima1, h =1, level = c(90, 95, 99))
  
  # --------------**********************GJR-GARCH**********************----------------------
  # provide the variance part of the traffic volume because of the conditional heteroscedastic
  #data(dmbp)
  # In page 5 of the paper, 
  # "Equation (10) indicates that the conditional distribution of et is independent and identically distributed
  # with zero mean and a variance of ht"
  # therefore the garch model is predict the mean of variance, that's why we are using "data = resid(arima1)^2" below
  # garch_pre@forecast$seriesFor is the mean of variance
  # garch_pre@forecast$sigmaFor is the standard deviation of variance
  # similarly in another paper, using kalman filter, we are also predicting the mean of variance 
  # and the variance of variance
  #--------------**************************************************-----------------

  spec1 <- ugarchspec(variance.model=list(model = "gjrGARCH",  garchOrder = c(1, 1)))
  gjrGARCHfit <- try(ugarchfit(data = resid(arima1)^2, spec = spec1), silent = TRUE) # notice here it is taking square
  # auto catch the error, don't throw it
  flag <- tryCatch(garch_pre <- ugarchforecast(gjrGARCHfit, n.ahead=1), error = c)
  
  # use the sigma from gjr-garch to calculate the prediction interval
  # http://faculty.washington.edu/ezivot/econ589/ch18-garch.pdf
  
  sd <- sqrt(garch_pre@forecast$seriesFor[1])
  garch_pre90_lo <- - 1.64*sd #garch_pre@forecast$seriesFor[1] 
  garch_pre90_hi <-  + 1.64*sd
  
  garch_pre95_lo <- - 1.96*sd
  garch_pre95_hi <-  + 1.96*sd
  
  garch_pre99_lo <-  - 2.58*sd
  garch_pre99_hi <-  + 2.58*sd
  
  # sum all the mean together
  #pre <- lr1_90[1] + ar_pre$mean[1] + garch_pre@forecast$seriesFor[1]
  pre <- lr1_90[1] + ar_pre$mean[1] 
  
  # the variance part is captured by the gjr_garch part which fits the residuals 
  # don't need consider the linear regression and arima
  pre_90lo <- pre  +  garch_pre90_lo #ar_pre$lower[1] +
  pre_90hi <- pre  +  garch_pre90_hi # + ar_pre$upper[1]
  
  pre_95lo <- pre  + garch_pre95_lo
  pre_95hi <- pre  + garch_pre95_hi
  
  pre_99lo <- pre  + garch_pre99_lo
  pre_99hi <- pre  + garch_pre99_hi
  
  res <- rbind(res, c(pre, pre_90lo, pre_90hi, pre_95lo, pre_95hi, pre_99lo, pre_99hi))
  i <- i + 1
  print (i)
}

write.csv(res, file = "Zhang_2014.csv")


