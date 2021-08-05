import matplotlib.pyplot as plt

from scipy.stats import poisson
lambda_ = 2
k=range(10)

pmf_X = poisson.pmf(k, lambda_)

fig, ax = plt.subplots(1, 1)
ax.bar(k, pmf_X)
plt.ylabel("pmf")
plt.xlabel("k")
plt.title("Probability mass function")
plt.show()