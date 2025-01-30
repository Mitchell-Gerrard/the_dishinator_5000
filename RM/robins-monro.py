# -*- coding: utf-8 -*-

import numpy as np
import math
from matplotlib import *
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import sdeint
# Noisy sine function
def noisy_sine(theta):
    return np.sin(theta) + np.random.normal(0,1)

# Simulate the Robbins-Monro algorithm for our
# noisy sine function.
# theta0 is the starting value, we perform 10000 iterations,
# and return the final output, which should be an approximation
# of a root of sin(theta).
# For instance, simulate_RM(1) has an overwhelming likelihood
# of returning something close to 0.
def simulate_RM(theta0,steps=10000):
    theta = theta0
    for N in range(1,steps+1):
        theta = theta - 1/N * noisy_sine(theta)
        if (abs(np.sin(theta)) < 0.01):
            return theta
    return theta

# theta0 is the starting value, thetahat is a root of sin(theta).
# By running 5000 trials, we estimate the probability of converging
# to the right thetahat given that we start at theta0.
def emperical_probability_RM(theta0,thetahat):
    total_trials = 0
    thetahat_trials = 0
    for i in range(0,5000):
        #result = simulate_RM(theta0)
        result = simulate_RM(theta0,steps=10000)

        total_trials += 1

        d = abs(thetahat - result) # distance from the root
        if (d < 0.1): # we will consider it having converged to the root if it is within 0.1
            thetahat_trials += 1
    return thetahat_trials / total_trials
def drift(t, theta):
    return -(1/t) * np.sin(theta)

# Define the diffusion (stochastic/noise) part of the SDE, scaled by 1/t
def diffusion(t, theta):
    return -1*np.random.normal(0,0.001)*1/t  # Scaling the noise by 1/t
def simulate_RM_sde(theta0,steps):
     t_span = np.linspace(1, steps, steps) 



     theta = sdeint.itoEuler(drift, diffusion,  theta0,t_span)
     plt.plot(t_span,theta)
     plt.show()
     return theta[-1]
def ode_RM(t,theta0,N):


    return -(1/t)*(np.sin(theta0)+np.random.normal(0,1))
def simulate_RM_ode(theta0,steps):
     t_eval = np.arange(1, steps+1, 1)

     sol = solve_ivp(ode_RM, [1, steps], [theta0], args=(steps,), t_eval=t_eval,rtol=0.01,atol=0.01,method='RK23')

     return(sol.y[0][-1])
# This is the best numerical fit I have found for the function
# emperical_probability_RM(theta0,thetahat)
def theoretical_probability_RM(theta0,thetahat):
    d = abs(theta0-thetahat)
    return 1 / (1 + math.exp(-3*(math.pi-d)))

# A couple of examples below where the numbers match well.
print("theta_0, theta^hat: " + str(3) + ", " + str(0))
print(theoretical_probability_RM(3,0))
print(emperical_probability_RM(3,0))

print("theta_0, theta^hat: " + str(2.4) + ", 2pi")
print(theoretical_probability_RM(2.4,2*math.pi))
print(emperical_probability_RM(2.4,2*math.pi))

print("theta_0, theta^hat: pi + 0.5, 2pi")
print(theoretical_probability_RM(math.pi+0.5,2*math.pi))
print(emperical_probability_RM(math.pi+0.5,2*math.pi))

# ... but it sadly seems to break at the endpoints.
print("theta_0, theta^hat: -2pi, 0")
print(theoretical_probability_RM(-2*math.pi,0))
print(emperical_probability_RM(-2*math.pi,0))
print("theta_0, theta^hat: 8pi, 0")
print(theoretical_probability_RM(8*math.pi,0))
print(emperical_probability_RM(7*math.pi,8*math.pi))
theta,theta_hat=np.linspace(0,np.pi,1000),[0]*1000
#for i,t in enumerate( theta):

#    per=emperical_probability_RM(t,theta_hat[i])

 #   plt.scatter(t,per)
#plt.show()
