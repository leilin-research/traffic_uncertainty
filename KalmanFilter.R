
require(dlm)

# time series of interest
TOI = read.csv("traffic_900.csv", header = FALSE)
TOI = ts(TOI)
plot(TOI)

train = TOI[1:600]
test = TOI[601:900]
# builds the model with parameters to be estimated
tsbuild15 = function(parm) {
  mod = dlmModPoly(order = 2, dV = exp(parm[1]), dW = c(exp(parm[2]),exp(parm[3]) )) + dlmModSeas(frequency = 15,dV = exp(parm[4]), dW = c(exp(parm[5]),rep(0,13)))
  mod$FF[1,1] = parm[6]
  mod$GG[1,1] = parm[7]
  mod$GG[1,2] = parm[8]
  mod$GG[2,2] = parm[9]
  return(mod)
}

y = train
#fit the model on the training set using maximum likelihood
fit15 = dlmMLE(y, c(rep(0,5),rep(1,4)), tsbuild15)
#predict on train + test
Pred15 = dlmFilter(TOI[1:900],tsbuild15(fit15$par))

#variance equation 2.8 b P53, Pred15$mod$F is 'H' Observation matrix
#dlmSvd2var(Pred15$U.R[[ii]],Pred15$D.R[ii,]) is 'Ppredicted', variance of error
#Pred15$mod$V is 'Q'
Q15 = vector(mode = "numeric", length = 900)
for(ii in 1:900)
  Q15[ii] = Pred15$mod$F %*% dlmSvd2var(Pred15$U.R[[ii]],Pred15$D.R[ii,])%*% t(Pred15$mod$F) + Pred15$mod$V

lower90 = Pred15$f[601:900]-1.645*sqrt(Q15)
upper90 = Pred15$f[601:900]+1.645*sqrt(Q15)

lower95 = Pred15$f[601:900]-1.96*sqrt(Q15)
upper95 = Pred15$f[601:900]+1.96*sqrt(Q15)

lower99 = Pred15$f[601:900]-2.575*sqrt(Q15)
upper99 = Pred15$f[601:900]+2.575*sqrt(Q15)

# 90% 1.645
# 95% 1.96
# 99% 2.575

write.csv(data.frame(601:900,lower90, upper90, lower95, upper95, lower99, upper99, predict = Pred15$f[601:900]), file ="Kalman.csv")
