# 1. Find the decision tree using ID3.
# Is it consistent with the training data (does it have 100% accuracy)?

import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
    

X = pd.DataFrame({'A': [1, 1, 0, 0],
                  'B': [1, 0, 1, 0],
                  'C': [0, 1, 1, 0]}) # Dataset variable

Y = pd.Series([0, 1, 1, 0]) # Target variable


# 1. Find the decision tree using ID3.
# Is it consistent with the training data (100% accuracy)

attributes = X[['A','B','C']]
target = Y.values
dt = tree.DecisionTreeClassifier(criterion='entropy').fit(X,Y)
fig, ax = plt.subplots(figsize=(7,8))
f = tree.plot_tree(dt, ax=ax, fontsize=8, feature_names=attributes.columns)
plt.show()

print('The accuracy for the set is: %f' %(dt.score(attributes, target)))

# We can see attributes A and B are not shown on the decision tree.
# A and B are not critical attributes
# Also, the tree is consistent with the training data (100% accuracy)

# 2. Is there a less deep decision tree consistent with the above data?
# If so, what logic concept does it represent?
# No, there is not because the entropy for the leaf nodes is 0