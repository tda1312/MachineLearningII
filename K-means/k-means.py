import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# load dataset
file_data = "../Datasets/Iris/iris.data"
df = pd.read_csv(file_data, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

# define features and target then separate the dataset
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
target = ['target']
x = df.loc[:, features].values
y = df.loc[:, target].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(x)

plt.scatter(x[:, 0], x[:, 1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title('K-means')
plt.show()