import sklearn.cluster
import numpy as np
import sklearn.metrics
import pandas as pd
import math
import itertools
from sklearn.metrics import confusion_matrix
import statistics
earth_radius = 6371.0

def calculate_metrics(df, location):
    # Filtra i valori teorici e predetti
    y_true = df['cluster_id'] == location
    y_pred = df['predicted_cluster'] == location

    # Calcolo della matrice di confusione
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    # Calcolo delle metriche
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity) if precision + sensitivity > 0 else 0

    return accuracy,precision,sensitivity,specificity, balanced_accuracy,F1
    

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

def degrees_to_meters(latitude, delta_lat=0, delta_lon=0):
    
    length_latitude_degree = (2 * math.pi * earth_radius*1000) / 360  # ~111.32 km in metri

    latitude_rad = math.radians(latitude)
    length_lon_degree = length_latitude_degree * math.cos(latitude_rad)

    meters_lat = delta_lat * length_latitude_degree
    meters_lon = delta_lon * length_lon_degree

    return meters_lat, meters_lon

original_tweets_dataset = pd.read_csv("tweets.csv")
tweets_dataset = original_tweets_dataset[['Latitude','Longitude']]
tweets_dataset = tweets_dataset.to_numpy()


theorical_eps_meters = 600
eps_rad = (theorical_eps_meters/1000 / earth_radius)

eps_dgs = math.degrees(eps_rad)

# Then using approximation of day and night hours estimate cluster of work and home
# evaluate single cluster parameters then aggregate them for the overall dataset, for each of the zones (home/work)

# Find number of unique users
users=original_tweets_dataset.value_counts(subset="User").index.to_numpy()

home_accuracies = []
precisions = []
sensitivities =[]
specificities = []
for u in users:
    # find all posts releated to one particular user and DBSCAN (varying the parameters)
    user_tweets = original_tweets_dataset[original_tweets_dataset['User'] == u]
    user_tweets_train = user_tweets[['Latitude','Longitude']]
    clustering = sklearn.cluster.DBSCAN(eps=eps_dgs,min_samples=4).fit(user_tweets_train)
    user_tweets['cluster'] = clustering.labels_
    unlabels = set(clustering.labels_)
    centroids_home = []
    centroids_work= []
    for i in unlabels:
        current_cluster = user_tweets[user_tweets['cluster'] == i]
        to_calculate = current_cluster[['Latitude','Longitude']].to_numpy()
        user_home_coords=current_cluster.iloc[0]['User_home']
        user_home_coords=user_home_coords[1:len(user_home_coords)-1]
        user_home_coords = user_home_coords.split(',')
        user_work_coords=current_cluster.iloc[0]['User_work']
        user_work_coords=user_work_coords[1:len(user_work_coords)-1]
        user_work_coords = user_work_coords.split(',')
        user_work_coords=np.array(user_work_coords,float)
        user_home_coords=np.array(user_home_coords,float)
        centroid_value=centroid(to_calculate)
        medoid_value=medoid(to_calculate)
        latitude_home = user_home_coords[0]
        latitude_work = user_work_coords[0]
        # print(f"{centroid_value}, home {user_home_coords}, work {user_work_coords}\n")
        # print(f"{medoid_value}, home {user_home_coords}, work {user_work_coords}\n")
        centroid_home=degrees_to_meters(latitude_home,*abs(centroid_value-user_home_coords))
        centroid_work=degrees_to_meters(latitude_work,*abs(centroid_value-user_work_coords))
        medoid_home=degrees_to_meters(latitude_home,*abs(medoid_value-user_home_coords))
        medoid_work=degrees_to_meters(latitude_work,*abs(medoid_value-user_work_coords))
        # print(f"Cluster: {i} user: {u}\n")
        # print(f"Difference between centroid, home {centroid_home} and work {centroid_work}\n")
        # print(f"Difference between medoid, home {medoid_home} and work {medoid_work}\n")
        centroids_home.append(sum(centroid_home))
        centroids_work.append(sum(centroid_work))
    home_cluster=np.argmin(centroids_home)
    work_cluster=np.argmin(centroids_work)
    user_tweets.loc[user_tweets['cluster']==home_cluster, 'predicted_cluster'] = 'home'
    user_tweets.loc[user_tweets['cluster']==work_cluster, 'predicted_cluster'] = 'work'
    user_tweets.loc[(user_tweets['cluster']!=work_cluster) & (user_tweets['cluster']!=home_cluster), 'predicted_cluster'] = 'outlier'
    home_accuracy,home_precision,home_sensitivity,home_specificity, home_balanced_accuracy,home_F1 = calculate_metrics(user_tweets,"home")
    work_accuracy,work_precision,work_sensitivity,work_specificity, work_balanced_accuracy,work_F1 = calculate_metrics(user_tweets,"work")
    home_accuracies.append(home_accuracy)
print(statistics.median(home_accuracies))
print(statistics.mean(home_accuracies))

