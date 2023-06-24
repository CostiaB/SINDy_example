

import torch
import torch.nn as nn
import torch.optim as optim



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))



class Encoder(nn.Module):
    def __init__(self, 
                 input_size : int,
                 hidden_dim : int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, hidden_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        out = torch.relu(self.fc5(x))
        return out
        
        

class Decoder(nn.Module):
    def __init__(self, 
                 input_size : int,
                 hidden_dim : int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64,128)
        self.fc4 = nn.Linear(128,256)
        self.fc5 = nn.Linear(256, input_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        out = torch.relu(self.fc5(x))
        return out
    
class Autoencoder(nn.Module):
    def __init__(self,
                 input_size : int,
                 hidden_dim : int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 ):
        super().__init__()
        self.encoder = encoder_class(input_size, hidden_dim)
        self.decoder = decoder_class(input_size, hidden_dim)
        
    def forward(self, x):
        dims = self.encoder(x)
        x_hat = self.decoder(dims)
        return x_hat
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    
    
    
    
    
    



'''
Data model
'''

import numpy as np

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
t, theta, omega = pendulum(initial_theta, initial_omega, pendulum_length, gravity, t_max=2)

X = theta
t1, X_test, omega1 = pendulum(initial_theta-0.2, initial_omega, pendulum_length, gravity, t_max=1)
width = 10

from scipy.linalg import hankel
data = hankel(X)
 
data1 = [ hankel(X[i:i+width]) for i in range(len(X))] 

test = [ hankel(X_test[i:i+width]) for i in range(len(X_test))] 

criterion = nn.MSELoss()
print(data.shape)




input_dim = 10
hidden_dim = int(input_dim * 1.5)
output_dim = input_dim

# Instantiate Fully-connected Autoencoder (FC-AE)
# And assign to model object
model = Autoencoder(input_size=input_dim,
                    hidden_dim=hidden_dim)

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

num_params_group = len(list(model.parameters()))
for group_idx in range(num_params_group):
    print(list(model.parameters())[group_idx].size())
    
    
    
    
    
num_epochs = 10    
idx = 0

# Dropout for creating noisy images
# by dropping out pixel with a 50% probability
dropout = nn.Dropout(0.5)



from tqdm import tqdm

mean_train, mean_test = [], []

for epoch in tqdm(range(num_epochs)):
    for i, mat in enumerate(data1[:-10]):
        total_train_loss = 0
        total_samples = 0

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output
        #mat = torch.from_numpy(mat)
        
        mat = torch.tensor(mat, dtype=torch.float)
        
       # mat = mat.double()
        outputs = model(mat)
        
        # Calculate Loss: MSE Loss based on pixel-to-pixel comparison
        loss = criterion(outputs, mat)

        # Getting gradients w.r.t. parameters via backpropagation
        loss.backward()

        # Updating parameters via gradient descent
        optimizer.step()
        
        total_samples += 1 #torch.numel(mat)
        total_train_loss += loss
        

        idx += 1

        if idx % 500 == 0:
            print(idx)
            # Calculate MSE Test Loss
            total_test_loss = 0
            total_samples = 0

            # Iterate through test dataset
            for test_mat in test[:-10]:
                
                test_mat = torch.tensor(test_mat, dtype=torch.float)
                

                # Forward pass only to get logits/output
                outputs = model(test_mat)

                # Test loss
                test_loss = criterion(outputs, test_mat)

                # Total number of labels
                total_samples += 1 #torch.numel(test_mat)

                # Total test loss
                total_test_loss += test_loss

            mean_test_loss = total_test_loss / total_samples
            mean_test.append(mean_test_loss)

            # Print Loss
            print(f'/nIteration: {idx}. Average Test Loss: {mean_test_loss.item()}.')

    mean_train_loss = total_train_loss / total_samples
    mean_train.append(mean_train_loss)
    print(f'/nIteration: {idx}. Average Test Loss: {mean_train_loss.item()}.')
        
        
import matplotlib.pyplot as plt
'''
mean_train = [t.detach().numpy() for t in mean_train]
mean_test = [t.detach().numpy() for t in mean_test]
plt.figure(figsize=(16,10))
plt.plot(mean_train)
plt.plot(mean_test) 
plt.show()       
'''      
