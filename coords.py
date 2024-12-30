import sklearn.cluster
import numpy as np
import sklearn.metrics
import pandas as pd
import math

earth_radius = 6371.0

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

clustering = sklearn.cluster.DBSCAN(eps=eps_dgs,min_samples=4).fit(tweets_dataset)
print(clustering.labels_)
original_tweets_dataset['cluster'] = clustering.labels_
print(original_tweets_dataset)
unlabels = set(clustering.labels_)

for i in unlabels:
    current_cluster = original_tweets_dataset[original_tweets_dataset['cluster'] == i]
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
    print(f"Cluster: {i}\n")
    print(f"Difference between centroid, home {degrees_to_meters(latitude_home,*abs(centroid_value-user_home_coords))} and work {degrees_to_meters(latitude_work,*abs(centroid_value-user_work_coords))}\n")
    print(f"Difference between medoid, home {degrees_to_meters(latitude_home,*abs(medoid_value-user_home_coords))} and work {degrees_to_meters(latitude_work,*abs(medoid_value-user_work_coords))}\n")