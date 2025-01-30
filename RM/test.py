import numpy as np
import sdeint
import matplotlib.pyplot as plt

# Define the drift (deterministic) part of the SDE
def drift(t, theta, N):
    return -(1/t) * np.sin(theta)  # Drift term: pulls toward the zeros of sin(theta)

# Define the diffusion (stochastic/noise) part of the SDE, scaled by 1/(1+t)
def diffusion(t, theta, N):
    # Uniform random noise between 0 and 1, much smaller noise to avoid large fluctuations
    noise = np.random.uniform(0, 0.01, size=theta.shape)  # Reduce the noise strength
    return noise * (1/(1 + t))  # Gradually decreasing noise over time

# Initial condition and parameters
theta0 = 1.0  # Initial condition, close to a zero of sin(theta)
N = 100  # Parameter N (doesn't directly influence drift or diffusion in this case)
steps = 10000  # Number of steps

# Ensure t_eval is a 1D array of evenly spaced time steps with smaller intervals
t_eval = np.linspace(1, steps, steps*10)  # Increase resolution for more accuracy

# Solve the SDE using the Euler-Maruyama method (Ito's method)
theta = sdeint.itoEuler(drift, diffusion, t_eval, theta0)

# Plot the solution
plt.plot(t_eval, theta)
plt.xlabel("Time")
plt.ylabel("Theta")
plt.title("Solution to Stochastic ODE using sdeint (Euler-Maruyama)")
plt.axhline(y=0, color='r', linestyle='--')  # Add a line at theta = 0 (a zero of sin)
plt.show()