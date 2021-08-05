from scipy.stats import uniform
from scipy.stats import norm
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np


def plot(x, interval):
    plt.style.use('ggplot')
    plt.hist(x)
    plt.show()

def exp_app(X):
    L = [np.prod([uniform.pdf(x) for x in X])]
    mle = max(L)

def analytical_app():
    # Step 1: Find the likelihood function
    # https://www.statology.org/mle-uniform-distribution/#:~:text=A%20uniform%20distribution%20is%20a,equally%20likely%20to%20be%20chosen.
    # Step 2: Apply the logarithm
    rcParams.update({'figure.autolayout': True})


w = 10
X = uniform.rvs(-w, w, size=100, random_state=1)
# mean, stdev = norm.fit(X)
# var = stdev**2

# 1. Plot the histogram of the data
plot(X, w)

# 2. Experimentally determine the MLE estimation for lambda given the observations in X
print(exp_app(X))
