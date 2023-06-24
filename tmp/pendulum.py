import numpy as np
import pysindy as ps

t = np.linspace(0, 1, 100)
x = 77 * np.exp(-2 * t)*np.random.uniform(0.8, 1.2, 1)


x = np.array([float(8 * np.exp(-2 * time) * np.random.uniform(0.99, 1.0, 1)) for time in t ])

y = 22 * np.exp(t)
X = np.stack((x, y), axis=-1)  # First column is x, second is y



model = ps.SINDy(feature_names=["x", "y"])
model.fit(X, t=t)


model.print()




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
t, theta, omega = pendulum(initial_theta, initial_omega, pendulum_length, gravity, t_max=10)

X = theta 


# Plot the results
plt.plot(t, theta)
plt.xlabel('Time')
plt.ylabel('Angle')
plt.title('Simple Pendulum Motion')
plt.grid(True)
#plt.show()

model = ps.SINDy(feature_names=["theta", "y"])
model.fit( theta , t=t)

print('Pendulum')
model.print()
print(model.get_feature_names())
m = model.simulate(x0=[1], t=np.arange(0.0, 0.50, 0.01) )
plt.plot(m)
plt.show()
def h_matrix(data, q):
    mat = []
    for t in range(len(data[:- q+1])):
        mat.append(data[t: t + q])
        
    return mat
q = 10
mat = h_matrix(theta, q)

from scipy.linalg import svd

U, s, Vh = svd(mat)

