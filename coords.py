import sklearn.cluster
import numpy as np
import sklearn.metrics

def centroid(data:list) -> list:
    medoid = [0 for i in range(len(data[0]))]
    for i in data:
        for index,elem in enumerate(i):
            medoid[index] += elem
    return [x/len(data) for x in medoid]

def medoid(data:list, func) -> list:
    #TODO to be implemented medoid (https://en.wikipedia.org/wiki/Medoid)
    pass

X = np.array([[1.0, 2.5], [2, 2], [2, 3],
              [8, 7], [8, 8], [3, 5]])

clustering = sklearn.cluster.DBSCAN(eps=2,min_samples=2).fit(X)
print(clustering.labels_)
set = set(clustering.labels_)
clusters = {}
for index in set:
    clusters[index] = []
    for x in zip(X,clustering.labels_):
        if(x[1] == index):
            clusters[index].append(x[0])
print(clusters)
