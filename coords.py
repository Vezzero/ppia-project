import sklearn.cluster
import numpy as np
import sklearn.metrics

def centroid(data:list) -> list:
    medoid = [0 for i in range(len(data[0]))]
    for i in data:
        for index,elem in enumerate(i):
            medoid[index] += elem
    return [x/len(data) for x in medoid]

def medoid(data: list) -> list:
    def distance(point1, point2):
        return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5
    best_medoid = None
    min_total_distance = float('inf')
    for candidate in data:
        total_distance = 0
        for point in data:
            total_distance += distance(candidate, point)
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_medoid = candidate
    return best_medoid

X = np.array([[1.0, 2.5], [2, 2], [2, 3],
              [8, 7], [8, 8], [3, 5]])

clustering = sklearn.cluster.DBSCAN(eps=2,min_samples=2).fit(X)
print(clustering.labels_)
labels = set(clustering.labels_)
clusters = {}
for index in set:
    clusters[index] = []
    for x in zip(X,clustering.labels_):
        if(x[1] == index):
            clusters[index].append(x[0])
print(clusters)
