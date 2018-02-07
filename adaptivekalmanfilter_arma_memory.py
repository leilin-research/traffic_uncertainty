# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:03:27 2016

@author: 20721659
"""

# -*- coding: utf-8 -*-

"""
basic functions are borrowed from
http://github.com/rlabbe/filterpy
"""

import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import scipy.linalg as linalg
from scipy.stats import multivariate_normal
import numpy.random as random
from numpy.random import randn
import pandas as pd


class KalmanFilter(object):
    """ Implements a Kalman filter. You are responsible for setting the
    various state variables to reasonable values; the defaults  will
    not give you a functional filter.
    You will have to set the following attributes after constructing this
    object for the filter to perform properly. Please note that there are
    various checks in place to ensure that you have made everything the
    'correct' size. However, it is possible to provide incorrectly sized
    arrays such that the linear algebra can not perform an operation.
    It can also fail silently - you can end up with matrices of a size that
    allows the linear algebra to work, but are the wrong shape for the problem
    you are trying to solve.
    Attributes
    ----------
    x : numpy.array(dim_x, 1)  --hidden state e.g., the parameters in time series function, 
        State estimate vector
    P : numpy.array(dim_x, dim_x) -- how much error will we get
        Covariance matrix
    R : numpy.array(dim_z, dim_z) -- z is observable/measurable, time series
        Measurement noise matrix
    Q : numpy.array(dim_x, dim_x)
        Process noise matrix
    A : numpy.array() -- A*x to get next x
        State Transition matrix
    H : numpy.array(dim_x, dim_x) -- H*x
        Measurement function
    You may read the following attributes.
    Attributes
    ----------
    y : numpy.array
        Residual of the update step.
    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step
    S :  numpy.array
        Systen uncertaintly projected to measurement space
    likelihood : scalar
        Likelihood of last measurement update.
    log_likelihood : scalar
        Log likelihood of last measurement update.
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        """ Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.
        Parameters
        ----------
        dim_x : int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.
            This is used to set the default size of P, Q, and u
        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.
        dim_u : int (optional)
            size of the control input, if it is being used.
            Default value of 0 indicates it is not used.
        """

        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0

        self.dim_x = dim_x # state variable
        self.dim_z = dim_z # measurement/observation variable
        self.dim_u = dim_u # control variable

        self.x = zeros((dim_x,1)) # state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.B = 0                # control transition matrix
        self.A = 0                # state transition matrix
        self.H = 0                 # Measurement function
        self.R = eye(dim_z)        # state uncertainty


        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0 # kalman gain
        self.y = zeros((dim_z, 1)) # residual
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty

        # identity matrix. Do not alter this.
        self.I = np.eye(dim_x)


    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.
        Parameters
        ----------
        z : np.array
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be a column vector.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        if z is None:
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H
        P = self.P
        x = self.x

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if x.ndim==1 and shape(z) == (1,1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        Hx = dot(H, x)
        if shape(Hx) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            Hx = np.asarray([Hx])

        assert shape(Hx) == shape(z) or (shape(Hx) == (1,1) and shape(z) == (1,)), \
               'shape of z should be {}, but it is {}'.format(
               shape(Hx), shape(z))
        self.y = z - Hx

        # S = HPH' + R
        # project system uncertainty into measurement space
        S = dot(H, dot(P, H.T)) + R
        
        if shape(S) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            S = np.asarray([S])

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        #K = dot(P, dot(H.T, linalg.inv(S)))
        K = dot(P, dot(H.T, 1/S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = x + dot(K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self.I - dot(K, H)
        self.P = dot(I_KH, dot(P, I_KH.T)) + dot(K, dot(R, K.T))

        self.S = S
        self.K = K

        # compute log likelihood
        mean = np.asarray(dot(H, x)).flatten()
        flatz = np.asarray(z).flatten()
        self.log_likelihood = multivariate_normal.logpdf(
             flatz, mean, cov=S, allow_singular=True)


    def predict(self, u=0, B=None, A=None, Q=None):
        """ Predict next position using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None in
            any position will cause the filter to use `self.B`.
        A : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None in
            any position will cause the filter to use `self.A`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None in
            any position will cause the filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if A is None:
            A = self.A
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Ax + Bu
        self.x = dot(A, self.x) + dot(B, u)

        # P = APA' + Q
        self.P = dot(A, dot(self.P, A.T)) + Q


    def get_prediction(self, u=0):
        """ Predicts the next state of the filter and returns it. Does not
        alter the state of the filter.
        Parameters
        ----------
        u : np.array
            optional control input
        Returns
        -------
        (x, P) : tuple
            State vector and covariance array of the prediction.
        """

        x = dot(self.A, self.x) + dot(self.B, u)
        P = dot(self.A, dot(self.A, self.A.T)) + self.Q
        return (x, P)


    def residual_of(self, z):
        """ returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        return z - dot(self.H, self.x)


    def measurement_of_state(self, x):
        """ Helper function that converts a state into a measurement.
        Parameters
        ----------
        x : np.array
            kalman state vector
        Returns
        -------
        z : np.array
            measurement corresponding to the given state
        """

        return dot(self.H, x)

class KalmanFilter_memory(object):
    """ Adaptive Kalman Filter approach for stochastic 
    short-term traffic flow rate prediction
    and uncertainty quantification
    ----------
    x : numpy.array(dim_x, 1)  --hidden state e.g., the parameters in time series function, 
        State estimate vector
    P : numpy.array(dim_x, dim_x) -- how much error will we get
        Covariance matrix
    R : numpy.array(dim_z, dim_z) -- z is observable/measurable, time series
        Measurement noise matrix
    Q : numpy.array(dim_x, dim_x)
        Process noise matrix
    A : numpy.array() -- A*x to get next x
        State Transition matrix
    H : numpy.array(dim_x, dim_x) -- H*x
        Measurement function
    You may read the following attributes.
    Attributes
    ----------
    y : numpy.array
        Residual of the update step.
    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step
    S :  numpy.array
        Systen uncertaintly projected to measurement space
    likelihood : scalar
        Likelihood of last measurement update.
    log_likelihood : scalar
        Log likelihood of last measurement update.
    """

    def __init__(self, dim_x, dim_z, N=10, dim_u=0):
        """ Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.
        Parameters
        ----------
        dim_x : int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.
            This is used to set the default size of P, Q, and u
        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.
        dim_u : int (optional)
            size of the control input, if it is being used.
            Default value of 0 indicates it is not used.
        
        N: memory length
        """

        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0

        self.dim_x = dim_x # state variable
        self.dim_z = dim_z # measurement/observation variable
        self.dim_u = dim_u # control variable

        self.x = zeros((dim_x,1)) # state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.B = 0                # control transition matrix
        self.A = 0                # state transition matrix
        self.H = 0                 # Measurement function
        self.R = eye(dim_z)        # state uncertainty
        self.N = N
        self.N_y = []           # historical errors, length not greater than N e=z-Hx
        self.N_P_prior = []
        self.N_P_poster = []
        self.N_a = []               # a = Xn-A*Xn-1
 
        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0 # kalman gain
        self.y = zeros((dim_z, 1)) # residual
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty

        # identity matrix. Do not alter this.
        self.I = np.eye(dim_x)


    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.
        Parameters
        ----------
        z : np.array
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be a column vector.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        if z is None:
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H
        P = self.P
        x = self.x

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if x.ndim==1 and shape(z) == (1,1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        Hx = dot(H, x)
        if shape(Hx) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            Hx = np.asarray([Hx])

        assert shape(Hx) == shape(z) or (shape(Hx) == (1,1) and shape(z) == (1,)), \
               'shape of z should be {}, but it is {}'.format(
               shape(Hx), shape(z))
        # step 2
        self.y = z - Hx
        if (len(self.N_y) == self.N):
            self.N_y.pop(0)
        self.N_y.append(self.y)
        
        #-- step 3 update R--------------------------
        R = 0
        ave_y = sum(self.N_y)/len(self.N_y)
        if shape(ave_y) == ():
                ave_y = np.asarray([ave_y])
                
        N = len(self.N_y)
        for i in range(N):
            his_y = self.N_y[i]
            his_P_prior = self.N_P_prior[i]
            if shape(his_y) == ():
                his_y = np.asarray([his_y])
            if shape(his_P_prior) == ():
                his_P_prior = np.asarray([his_P_prior])
            R = R + dot(his_y - ave_y, (his_y-ave_y).T)-(N-1)/N*dot(self.H, dot(his_P_prior, self.H.T))
        R = R/N
        self.R = R
        
        # step 4
        # S = HPH' + R
        # project system uncertainty into measurement space
        S = dot(H, dot(P, H.T)) + R
        
        if shape(S) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            S = np.asarray([S])

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        #K = dot(P, dot(H.T, linalg.inv(S)))
        K = dot(P, dot(H.T, 1/S))
        
        # step 5
        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = x + dot(K, self.y)
        
        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self.I - dot(K, H)
        self.P = dot(I_KH, dot(P, I_KH.T)) + dot(K, dot(R, K.T))
        
        Ptem = eye(self.dim_x) * 0 # because in step (7), need two Poster Ps, when index is 0, need the P at -1
        if (len(self.N_P_poster) == self.N):
            Ptem = self.N_P_poster.pop(0)
        self.N_P_poster.append(self.P)
        
        self.S = S
        self.K = K
        
        # step 6
        a = self.x - dot(self.A, x)
        if (len(self.N_a) == self.N):
            self.N_a.pop(0)
        self.N_a.append(a)
        # step 7
        ave_a = sum(self.N_a)/len(self.N_a)
        
        Q = eye(self.dim_x) * 0
        N = len(self.N_a)
        for i in range(N):
            his_a = self.N_a[i]
            his_P = self.N_P_poster[i]
            if i == 0:
                his_P1 = Ptem
            else:
                his_P1 = self.N_P_poster[i-1]
            if shape(his_a) == ():
                his_a = np.asarray([his_a])
            if shape(his_P) == ():
                his_P = np.asarray([his_P])
            if shape(his_P1) == ():
                his_P1 = np.asarray([his_P1])
                
            Q = Q + dot(his_a - ave_a, (his_a-ave_a).T)-(N-1)/N*(dot(self.A, dot(his_P1, self.A.T)) - his_P)
            
        Q = Q/N
        self.Q = Q
        # compute log likelihood
        #mean = np.asarray(dot(H, x)).flatten()
        #flatz = np.asarray(z).flatten()
        #self.log_likelihood = multivariate_normal.logpdf(
        #     flatz, mean, cov=S, allow_singular=True)


    def predict(self, u=0, B=None, A=None, Q=None):
        """ Predict next position using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None in
            any position will cause the filter to use `self.B`.
        A : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None in
            any position will cause the filter to use `self.A`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None in
            any position will cause the filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if A is None:
            A = self.A
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # step 1
        # x = Ax + Bu
        self.x = dot(A, self.x) + dot(B, u)

        # P = APA' + Q
        self.P = dot(A, dot(self.P, A.T)) + Q
        
        if (len(self.N_P_prior) == self.N):
            self.N_P_prior.pop(0)
        self.N_P_prior.append(self.P)


    def get_prediction(self, u=0):
        """ Predicts the next state of the filter and returns it. Does not
        alter the state of the filter.
        Parameters
        ----------
        u : np.array
            optional control input
        Returns
        -------
        (x, P) : tuple
            State vector and covariance array of the prediction.
        """

        x = dot(self.A, self.x) + dot(self.B, u)
        P = dot(self.A, dot(self.A, self.A.T)) + self.Q
        return (x, P)


    def residual_of(self, z):
        """ returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        return z - dot(self.H, self.x)


    def measurement_of_state(self, x):
        """ Helper function that converts a state into a measurement.
        Parameters
        ----------
        x : np.array
            kalman state vector
        Returns
        -------
        z : np.array
            measurement corresponding to the given state
        """

        return dot(self.H, x)
        
if __name__ == "__main__":
    
    f = KalmanFilter_memory(dim_x=1, dim_z=1, N=100) # change the memory size here

    f.x = np.array([50])      

    f.A = np.array([0.9127])    # state transition matrix

    f.H = np.array([0.0571])    # Measurement function
    f.P *= 1000                  # covariance matrix
    f.R = 212.72                      # state uncertainty
    f.Q = 707858.9                # process uncertainty

    measurements = []
    results = []

    zs = []
    
    z = pd.read_csv('kf_arima.csv', header = 0)
    z=z['x']
    
    for t in range (len(z)):
        # create measurement = t plus white noise
        # z = t + random.randn()*20
        zs.append(np.array(z[t]))

        # perform kalman filtering
        f.predict()
        f.update(z[t])

        # save data
        results.append(f.x * 0.0571)  # remeber to change f.H here
        measurements.append(z[t])
    
    res = pd.DataFrame(results)
    res.to_csv('kf_arima_predicted_N_100.csv')