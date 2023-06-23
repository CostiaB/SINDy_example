# SINDy_example
Sparse Identification of Nonlinear Dynamical Systems (SINDy) is an algorithm to discover governing dynamical equations for time series  𝐱(𝑡) .

The key idea is to construct a differential equation:

$$\frac{d{\bf x}}{d t} = \Theta({\bf x}^T) {\bf x}$$
or like this:
$$\frac{d{\bf x}}{d t^2} = \Theta({\bf x}^T) {\bf x}$$
if we are looking for a second-order derivative.

where the time derivative is computed from the time series data and Θ(𝐱) is a non-linear basis.

We will follow the next steps:
Steps:

Compute time derivative: Perhaps the trickiest part, especially for noisy time series although one can use total variation regularized derivatives for such cases as here
  
Choose basis: We will choose a non-linear basis constructed from the time series data, consisting of a polynomial basis and trigonometrical functions.  
  
Use regression: We apply regularized linear regression Lasso/Ridge with one step or we can perform sequential thresholding least squares.
  
Paper: https://www.pnas.org/content/113/15/3932

Once the underlying dynamical equations are discovered, forecasting becomes a lot easier.

Extensions:

Knowing the right coordinates and basis functions to use is often difficult. The most interesting extension of SINDy so far has been to use latent basis / coordinates instead of physical space basis. In latent space,


$$ {\bf x} \longrightarrow {\bf z} = Encoder({\bf x}).$$

A non-linear basis in 𝐳 is used to perform SINDy. The full cost function also takes into account the physical space SINDy expression. 
