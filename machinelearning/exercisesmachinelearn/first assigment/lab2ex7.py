# In order to finish a board game, a player must get an exact 3 on a regular die.
# Using the scipy.stats.geom package, determine:
# how many tries will it take to win the game (on average)?
# What are the best and worst cases?

from scipy.stats import geom

THRESHOLD = 0.01 # Trust level
p = 1/6 # The probability of getting an exact number
k = 1 # Number of tries
f = geom.pmf(k,p) # Probability of k failures before success

while f > THRESHOLD:
    k+=1
    f = geom.pmf(k,p)

print('Average tries: %d' %(k)) # Average number of tries
print('Best and worst number of tries for a fair die with probability %f :' %(p))
print(geom.interval(0.95,p)) # Best and worst cases