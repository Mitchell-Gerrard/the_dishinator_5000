# -*- coding: utf-8 -*-

from numpy import *
import math
from matplotlib import *
import matplotlib


# Noisy sine function
def noisy_sine(theta):
    return math.sin(theta) + random.normal(0,1)

# Simulate the Robbins-Monro algorithm for our
# noisy sine function.
# theta0 is the starting value, we perform 10000 iterations,
# and return the final output, which should be an approximation
# of a root of sin(theta).
# For instance, simulate_RM(1) has an overwhelming likelihood
# of returning something close to 0.
def simulate_RM(theta0):
    theta = theta0
    for N in range(1,10000):
        theta = theta - 1/N * noisy_sine(theta)
    return theta

# theta0 is the starting value, thetahat is a root of sin(theta).
# By running 5000 trials, we estimate the probability of converging
# to the right thetahat given that we start at theta0.
def emperical_probability_RM(theta0,thetahat):
    total_trials = 0
    thetahat_trials = 0
    for i in range(0,5000):
        result = simulate_RM(theta0)
        total_trials += 1
        d = abs(thetahat - result) # distance from the root
        if (d < 0.1): # we will consider it having converged to the root if it is within 0.1
            thetahat_trials += 1
    return thetahat_trials / total_trials

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
