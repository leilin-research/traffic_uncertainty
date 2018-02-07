# This repository includes the codes and data for:

the paper "Quantifying Uncertainty in Short-term Traffic Prediction and its Application to Optimal Staffing Plan Development" submitted to Transportation Research Record Part C: Emerging Technologies. 

**Model:**

pso.m - particle swarm optimization algorithm

elm_pi.m - extreme learning machine algorithm to provide prediction intervals

pso_elm.m - optimize elm parameters with pso for interval predictions

Improved_pso_elm.m - on-line version of pso-elm model


**Results:**

ARMA.csv - results using ARMA model

Zhang_2014.csv - results based on paper "Zhang, Y., Zhang, Y., Haghani, A., 2014. A hybrid short-term traffic flow forecasting meth-od based on spectral analysis and statistical volatility model."

Kalman.csv - results using Kalman Filter model

Guo_2014.csv - results based on paper "Guo, J., Huang, W., Williams, B.M., 2014. Adaptive Kalman filter approach for stochastic short-term traffic flow rate prediction and uncertainty quantification"

PSO_ELM.csv - results using PSO-ELM model

Improved_PSO_ELM.csv - results using Improved PSO-ELM model

