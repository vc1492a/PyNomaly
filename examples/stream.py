import numpy as np
from PyNomaly import loop
import pandas as pd
from pydataset import data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


iris = pd.DataFrame(data('iris'))
iris = pd.DataFrame(iris.drop('Species', 1))

iris = iris.sample(frac=1) # shuffle data
iris_train = iris.iloc[:, 0:4].head(120)
iris_test = iris.iloc[:, 0:4].tail(30)

m = loop.LocalOutlierProbability(iris).fit()
scores_noclust = m.local_outlier_probabilities
iris['scores'] = scores_noclust

m_train = loop.LocalOutlierProbability(iris_train, n_neighbors=10)
m_train.fit()
iris_train_scores = m_train.local_outlier_probabilities

iris_test_scores = []
for index, row in iris_test.iterrows():
    array = np.array([row['Sepal.Length'], row['Sepal.Width'], row['Petal.Length'], row['Petal.Width']])
    iris_test_scores.append(m_train.stream(array))
iris_test_scores = np.array(iris_test_scores)

iris['stream_scores'] = np.hstack((iris_train_scores, iris_test_scores))
# iris['scores'] from earlier example
rmse = np.sqrt(((iris['scores'] - iris['stream_scores']) ** 2).mean(axis=None))
print(rmse)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iris['Sepal.Width'], iris['Petal.Width'], iris['Sepal.Length'],
c=iris['stream_scores'], cmap='seismic', s=50)
ax.set_xlabel('Sepal.Width')
ax.set_ylabel('Petal.Width')
ax.set_zlabel('Sepal.Length')
plt.show()
plt.clf()
plt.cla()
plt.close()
