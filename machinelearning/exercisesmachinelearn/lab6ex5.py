# Homework for lab 6

from re import X
from numpy.lib.function_base import blackman
from scipy.stats import bernoulli
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pandas as pd


p = 0.5
size = 1000
a = np.zeros(1000, dtype=np.int) # If we try np.full the bernoulli.rvs(p) value is repeated 1000 times
b = np.zeros(1000, dtype=np.int)
c = np.zeros(1000, dtype=np.int)
y = np.zeros(1000, dtype=np.int)

for i in range(len(a)): 
    a[i] = bernoulli.rvs(p)
    b[i] = bernoulli.rvs(p)
    c[i] = bernoulli.rvs(p)
    y[i] = (a[i] and b[i]) or ( not(b[i] or c[i]) )

d = pd.DataFrame(
    {
        'A': a,
        'B': b,
        'C': c,
        'Y': y
    }
)

X = d[['A', 'B', 'C']]
Y = d['Y']
cl = BernoulliNB().fit(X, Y)

# 1. Generate a DataFrame with 1000 entries and four columns A, B, C and Y,
# according to the description above, using the bernoulli.rvs function from scipy.
print("The dataframe for the function Y = (A and B) or not(B or C) : ")
print(d)

# 2. Calculate the error rate on the training dataset.
print("Error rate on the training dataset: %f" % (1- cl.score(X,Y)))

# 3. What is the average error rate on this training dataset for the Joint Bayes algorithm?
# (Note that you don't have to actually build the algorithm, just provide a theoretical justification.)
# The value for each variable is either 1 or 0, therefore there are 2^3 combinations
# There are 4 combinations which meet the criteria so the error rate is 1 - 4/9 = 0.56