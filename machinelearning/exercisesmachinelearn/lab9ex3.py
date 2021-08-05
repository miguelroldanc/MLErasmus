from scipy.stats import norm
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut 
from sklearn.model_selection import cross_val_score
from statistics import mean

x_red = norm.rvs(0,1,100,random_state=1)
y_red = norm.rvs(0,1,100,random_state=2)
x_green = norm.rvs(1,1,100,random_state=3)
y_green = norm.rvs(1,1,100,random_state=4)
d = pd.DataFrame({
    'X1': np.concatenate([x_red,x_green]),
    'X2': np.concatenate([y_red, y_green]),
    'Y': [1]*100+[0]*100
})

X, Y = d[['X1', 'X2']], d['Y']


# 1. Plot the dataset using pyplot
c = ['green' if l == 0 else 'red' for l in Y]
fig, ax = plt.subplots(figsize=(5,5))
plt.scatter(X['X1'], X['X2'], color=c)
plt.show()


# 2. Compare the training error of the Adaboost algorithm 
# (using the usual decision stumps as weak learners)
# and the ID3 algorithm
id3 = DecisionTreeClassifier(criterion='entropy').fit(X,Y)
print('The depth of ID3 complete tree is %d' % (id3.get_depth()))
print('The number of max iterators for AdaBoost is %d' % (id3.get_depth()*30))

# Compute the score
ab_score = []
id3_score = []
k_all = range(1,15)
for i in k_all:
    ab = AdaBoostClassifier(n_estimators=30*i).fit(X,Y)
    ab_score.append(ab.score(X, Y))
    id3 = DecisionTreeClassifier(criterion='entropy',max_depth=i).fit(X,Y)
    id3_score.append(id3.score(X,Y))

# Plot the score
fig, ax = plt.subplots(figsize=(5,5))
plt.plot(k_all, ab_score, color='blue', label='AdaBoost')
plt.plot(k_all, id3_score, color='green', label='ID3')
plt.title('Score comparative between AdaBoost and ID3')
plt.legend()
plt.show()


# 3. Compare the CVLOO error of the AdaBoost algorithm 
# (using the usual decision stumps as weak learners) 
# and the ID3 algorithm
loo = LeaveOneOut()
ab_score = []
id3_score = []

start_time = time.time()
for i in k_all:
    ab = AdaBoostClassifier(n_estimators=30*i).fit(X,Y)
    ab_score.append( mean(cross_val_score(ab, X, Y, cv=loo)) )
    id3 = DecisionTreeClassifier(criterion='entropy',max_depth=i).fit(X,Y)
    id3_score.append( mean(cross_val_score(id3, X, Y, cv=loo)) )

print("--- %s seconds ---" % (time.time() - start_time))
# Plot the score
fig, ax = plt.subplots(figsize=(5,5))
plt.plot(k_all, ab_score, color='blue', label='AdaBoost')
plt.plot(k_all, id3_score, color='green', label='ID3')
plt.title('Score comparative between AdaBoost and ID3 (CVLOO version)')
plt.legend()
plt.show()