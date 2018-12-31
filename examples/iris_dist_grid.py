from PyNomaly import loop
import pandas as pd
from pydataset import data
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


iris = pd.DataFrame(data('iris'))
iris = pd.DataFrame(iris.drop('Species', 1))

distance_metrics = [
    'braycurtis',
    'canberra',
    'cityblock',
    'chebyshev',
    'cosine',
    'euclidean',
    'hamming',
    'l1',
    'manhattan'
]

fig = plt.figure(figsize=(17, 17))

for i in range(1, 10):

    neigh = NearestNeighbors(n_neighbors=10, metric=distance_metrics[i-1])
    neigh.fit(iris)
    d, idx = neigh.kneighbors(iris, return_distance=True)

    m = loop.LocalOutlierProbability(distance_matrix=d,
                                     neighbor_matrix=idx).fit()
    iris['scores'] = m.local_outlier_probabilities

    ax = fig.add_subplot(3, 3, i, projection='3d')
    plt.title(distance_metrics[i-1], loc='left', fontsize=18)
    ax.scatter(iris['Sepal.Width'], iris['Petal.Width'], iris['Sepal.Length'],
               c=iris['scores'], cmap='seismic', s=50)
    ax.set_xlabel('Sepal.Width')
    ax.set_ylabel('Petal.Width')
    ax.set_zlabel('Sepal.Length')


plt.show()
plt.clf()
plt.cla()
plt.close()

