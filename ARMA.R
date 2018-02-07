data1<-read.csv('traffic_900.csv');
library(forecast);
train<-data1[1:600,1];
test<-data1[601:900,1];


# off line training
sensor<-ts(train,frequency=15);
fit <- auto.arima(sensor,approximation=FALSE,trace=FALSE);
fcast <- forecast(fit,h=300,level=c(80,95));# typeof(fcast)
res <- data.frame(fcast[[4]],fcast[[5]],fcast[[6]]);#transfer time series back to data frame
write.csv(res, "ARMA_res.csv") 
# the five columns are point predictions, lower bound 80, lower bound 95, upper bound 80, upper bound 95






