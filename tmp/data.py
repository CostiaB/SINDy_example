import numpy as np
import matplotlib.pyplot as plt

def pendulum(theta, omega, L, g, t_max = 10.0):
    """
    Simulates the motion of a simple pendulum.
    Equation theta'' = -(g/l) * sin(theta)

    Parameters:
        theta (float): Initial angle in radians.
        omega (float): Initial angular velocity in radians per second.
        L (float): Length of the pendulum string.
        g (float): Acceleration due to gravity.
        t_max (float): Maximum simulation time

    Returns:
        Tuple: Arrays of time, angle, and angular velocity values.
    """
    # Simulation parameters
    dt = 0.01  # Time step

    # Initialize arrays
    t_values = np.arange(0.0, t_max, dt)
    theta_values = np.zeros_like(t_values)
    omega_values = np.zeros_like(t_values)

    # Set initial conditions
    theta_values[0] = theta
    omega_values[0] = omega

    # Perform simulation
    for i in range(1, len(t_values)):
        # Calculate derivatives
        theta_dot = omega_values[i - 1]
        omega_dot = -(g / L) * np.sin(theta_values[i - 1])

        # Update variables using Euler's method
        theta_values[i] = theta_values[i - 1] + theta_dot * dt
        omega_values[i] = omega_values[i - 1] + omega_dot * dt

    return t_values, theta_values, omega_values

# Define initial conditions and parameters
initial_theta = np.pi / 4  # Initial angle (45 degrees)
initial_omega = 0.0  # Initial angular velocity
pendulum_length = 10  # Length of the pendulum string
gravity = 9.8  # Acceleration due to gravity

# Simulate the pendulum motion
t, theta, omega = pendulum(initial_theta, initial_omega, pendulum_length, gravity, t_max=150)

X = theta 