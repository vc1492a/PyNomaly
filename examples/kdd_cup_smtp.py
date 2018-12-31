from PyNomaly import loop
from sklearn.datasets import fetch_kddcup99

data = fetch_kddcup99(subset='smtp', percent10=False)
train_data, target = data.data, data.target
m = loop.LocalOutlierProbability(train_data[0:1000].astype(float), extent=3, n_neighbors=6).fit()


