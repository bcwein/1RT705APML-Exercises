import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats import multivariate_normal

# load data from disk
data = scipy.io.loadmat("lindata.mat")
x = data["X"]  # inputs
y = data["Y"]  # outputs
beta = float(data["sigma"])**(-2)  # measurement noise precision
N = len(x)  # number of data points

# Define the feature vector
def Phi(a):  # Phi(a) = [1,a]
    return np.power(a, range(2))

# then, define the prior
D = len(Phi(0))  # number of features

# set parameters of prior on the weights
mu0 = np.zeros((D, 1))
Sigma0 = 10*np.eye(D) / D  # p(w)=N(mu0,Sigma0)

# Do the regression
SN = np.linalg.inv(np.linalg.inv(Sigma0) + beta * Phi(x).T @ Phi(x))
mN = SN @ (np.linalg.inv(Sigma0) @ mu0 + beta * Phi(x).T @ y)

n = 100  # number of grid-points, for plotting
xs = np.linspace(-8, 8, n)[:, np.newaxis]  # reshape is needed for Phi to work
m = Phi(xs) @ mu0 

# Compute the postrior distribution of the function
mpost = Phi(xs) @ mN 
vpost = Phi(xs) @ SN @ Phi(xs).T

# Compute the predictive distribution of the outputs
mpred = mpost
vpred = vpost + beta**(-1)

s = 5 # Draw samples from the posterior
fs = multivariate_normal(mean=mpost.flatten(),cov=vpost,allow_singular=True).rvs(s).T


### Code for the plotting
plt.plot(xs,Phi(xs)) # Plot the features
plt.title('features')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


plt.plot(xs,fs,'gray') # Plot the samples
plt.scatter(x,y,zorder=3)
plt.title('posterior - samples')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Reshape (plt.fill_between requires a one-dimensional array)
xsf = xs.flatten()
mpredf = mpred.flatten()
stdpred = np.sqrt(np.diag(vpred))


plt.plot(xs,mpred,'black') # Plot credibility regions
plt.fill_between(xsf,mpredf + 3*stdpred,mpredf - 3*stdpred,color='lightgray')
plt.fill_between(xsf,mpredf + 2*stdpred,mpredf - 2*stdpred,color='darkgray')
plt.fill_between(xsf,mpredf + 1*stdpred,mpredf - 1*stdpred,color='gray')
plt.scatter(x,y,zorder=3)
plt.title('predictive distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()