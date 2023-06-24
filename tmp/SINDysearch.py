import torch
import itertools

class SINDysearch():
    def __init__(self,
                 latent_dim=3,
                 include_biases=True,
                 include_states=True,
                 include_sin=False,
                 include_cos=True,
                 include_multiply_pairs=False,
                 poly_order=2,
                 include_sqrt=True,
                 include_inverse=True,
                 include_sign_sqrt_of_diff=False,
                 device='cuda:0'):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.candidate_functions = []
        self.feature_names = []
        # initialize lib with biasses 
        self.latent_dim = latent_dim

        self.include_biases = include_biases
        self.include_states = include_states
        self.include_sin = include_sin
        self.include_cos = include_cos
        self.include_multiply_pairs=include_multiply_pairs
        self.include_inverse = include_inverse
        self.poly_order = poly_order
        self.include_sqrt = include_sqrt
        self.include_singn_sqrt_of_diff = include_sign_sqrt_of_diff

        # fit for functions and feature names
        self.fit()
        self.number_candidate_functions = len(self.feature_names)
        

    def biases(self, z):
        return torch.ones(z.shape[0], 1, device=self.device)

    @staticmethod
    def states(z):
        return z

    @staticmethod
    def sin(z):
        return - torch.sin(z)
    
    @staticmethod
    def cos(z):
        return torch.cos(z)

    def multiply_pairs(self, z):
        result = []
        for idx1, idx2 in self.idx_combis_commutative:
            res = z[:, idx1] * z[:, idx2]
            res = res.reshape(-1, 1)
            result.append(res)
        return torch.cat(result, axis=1)

    @staticmethod
    def inverse(z):
        return 1/z

    def poly_deg_2(self, z):
        result = []
        for i in range(self.latent_dim):
            for j in range(i, self.latent_dim):
                    res = z[:,i]*z[:,j]
                    res = res.reshape(-1, 1)
                    result.append(res)
        return torch.cat(result, axis=1)

    @staticmethod
    def sqrt(z):
        return torch.sqrt(z)

    def sing_sqrt_diff_pairs(self, z):
        result = []
        for idx1, idx2 in self.idx_combis_commutative:
            res = torch.sign(z[:, idx1] - z[:,idx2])*torch.sqrt(torch.abs(z[:, idx1] - z[:,idx2]))
            res = res.reshape(-1, 1)
            result.append(res)
        return torch.cat(result, axis=1)



    def fit(self):
        ## generate all possible pairs of z variable indices
        possible_indicies = list(range(self.latent_dim))
        permuts = itertools.product(possible_indicies, possible_indicies)
        permuts = [p for p in permuts if not p[0] == p[1]]
        self.idx_combis_non_commutative = permuts
        self.idx_combis_commutative = list(set([tuple(sorted(list(p))) for p in permuts]))

        if self.include_biases:
            self.candidate_functions.append(self.biases)
            names = ['1']
            self.feature_names.extend(names)
        if self.include_states:
            self.candidate_functions.append(self.states)
            names = [f'z{i}' for i in range(self.latent_dim)]
            self.feature_names.extend(names)
        if self.include_sin:
            self.candidate_functions.append(self.sin)
            names = [f'sin(z{i})' for i in range(self.latent_dim)]
            self.feature_names.extend(names)
        if self.include_cos:
            self.candidate_functions.append(self.cos)
            names = [f'cos(z{i})' for i in range(self.latent_dim)]
            self.feature_names.extend(names)
        if self.include_multiply_pairs: 
            self.candidate_functions.append(self.multiply_pairs)
            names = [f'z{idx1}*z{idx2}' for idx1, idx2 in self.idx_combis_commutative]
            self.feature_names.extend(names)
        if self.include_inverse:
            self.candidate_functions.append(self.inverse)
            names = [f'1/z{i}'for i in range(self.latent_dim)]
            self.feature_names.extend(names)
        if self.poly_order == 2:
            self.candidate_functions.append(self.poly_deg_2)
            names = []
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                        names.append(f'z{i}*z{j}')
            self.feature_names.extend(names)
        if self.include_sqrt:
            self.candidate_functions.append(self.sqrt)
            names = [f'sqrt(z{i})' for i in range(self.latent_dim)]
            self.feature_names.extend(names)
        if self.include_singn_sqrt_of_diff:
            self.candidate_functions.append(self.sing_sqrt_diff_pairs)
            names = [f'sign(z{idx1}-z{idx2})*sqrt(|z{idx1}-z{idx2}|)'
                     for idx1, idx2 in self.idx_combis_commutative]
            self.feature_names.extend(names)

    def get_feature_names(self,):
        return self.feature_names
    
    def transform(self, z):
      trans = [cand_func(z) for cand_func in self.candidate_functions]
      out =  torch.cat(trans, axis=1)
      return out
'''
if __name__ == '__main__':
    # some test for the SINDy lib
    
    z = torch.tensor([[1, 2, 3], [4, 0, 6]])
    sl = SINDysearch()
    theta = sl.transform(z)
'''

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
t, theta, omega = pendulum(initial_theta, initial_omega, pendulum_length, gravity, t_max=15)


def h_matrix(data, q):
    mat = []
    for t in range(len(data[:- q+1])):
        mat.append(data[t: t + q])
        
    return mat
q = 10
X = h_matrix(theta, q)

X = torch.tensor(X, dtype=torch.float)

sl = SINDysearch()
ang = sl.transform(X)

sl.fit()
print(sl.get_feature_names())












