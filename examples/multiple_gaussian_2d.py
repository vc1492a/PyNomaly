import numpy as np
import matplotlib.pyplot as plt
from PyNomaly import loop
import pandas as pd
from sklearn import preprocessing


# import the multiple gaussian data #
df = pd.read_csv('../data/multiple-gaussian-2d-data-only.csv')
print(df)

# fit LoOP according to the original settings outlined in the paper #
m = loop.LocalOutlierProbability(df[['x', 'y']], n_neighbors=20, extent=3).fit()
scores = m.local_outlier_probabilities
print(scores)

# plot the results #
# base 3 width, then set as multiple
threshold = 0.1
color = np.where(scores > threshold, "white", "black")
label_mask = np.where(scores > threshold)
area = (20 * scores) ** 2
plt.scatter(df['x'], df['y'], c=color, s=list(area), edgecolor='red', linewidth='1')
plt.scatter(df['x'], df['y'], c='black', s=3)
for i in range(len(scores)):
    if scores[i] > threshold:
        plt.text(df['x'].loc[i] * (1 + 0.01), df['y'].loc[i] * (1 + 0.01), round(scores[i], 2), fontsize=8)

plt.show()


'''
The above plot shows that the results from this implementation generally 
match in ranking vs the results shown in the original paper, but the 
probabilities are off by some margin in some cases. 
Examination points to the calculation of the expected values during the 
estimation process: where we calculate expected values using global means, 
the paper mentions calculating the expected value over the neighborhood 
about a particular observation. 
'''


