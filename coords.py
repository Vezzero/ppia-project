import sklearn.cluster
import numpy as np
import sklearn.metrics
import pandas as pd
import math
import itertools
from sklearn.metrics import confusion_matrix
import statistics
from scipy.spatial import Voronoi, voronoi_plot_2d
pd.options.mode.chained_assignment = None
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

def assign_voronoi_clusters(df, centers):
    # vonoroi diag
    vor = Voronoi(centers)
    #For each point, find nearest Voronoi seed
    #The seeds for the Voronoi diagram are the points in centers
    points = df[['Latitude','Longitude']].to_numpy()
    labels = []
    
    for p in points:
        dists = np.linalg.norm(centers - p, axis=1)
        labels.append(np.argmin(dists))

    return labels, vor

# Find number of unique users
def analyze(dataset_path_without_csv,theorical_eps_meters):
    original_tweets_dataset = pd.read_csv(f"{dataset_path_without_csv}.csv")
    tweets_dataset = original_tweets_dataset[['Latitude','Longitude']]
    tweets_dataset = tweets_dataset.to_numpy()

    #min_samples_values = [i for i in range(4,round(math.sqrt(len(tweets_dataset))),round(math.sqrt(len(tweets_dataset))/20))]
    min_samples_values=[i for i in range(4,100,5)]
    users=original_tweets_dataset.value_counts(subset="User").index.to_numpy()

    results = []
    for eps in theorical_eps_meters:
        eps_dgs=math.degrees((eps / 1000) / earth_radius)
        for min_samples in min_samples_values:
            accuracies = [[],[]]
            precisions = [[],[]]
            sensitivities =[[],[]]
            specificities = [[],[]]
            balanced_accuracies = [[],[]]
            F1_scores = [[],[]]
            centroid_distances = [[],[]]
            for u in users:
            # find all posts releated to one particular user and DBSCAN (varying the parameters)
                user_tweets = original_tweets_dataset[original_tweets_dataset['User'] == u]
                user_tweets_train = user_tweets[['Latitude','Longitude']]
                clustering = sklearn.cluster.DBSCAN(eps=eps_dgs, min_samples=min_samples).fit(user_tweets_train)
                user_tweets['cluster'] = clustering.labels_
                unlabels = set(clustering.labels_)
                unlabels = set(clustering.labels_)
                unlabels = {label for label in unlabels if label >= 0}
                if not unlabels:
                    continue
                centroids_home = []
                centroids_work= []
                cluster_centers = []
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

                    try:
                        user_home_coords = np.array(eval(current_cluster.iloc[0]['User_home']), dtype=float)
                        user_work_coords = np.array(eval(current_cluster.iloc[0]['User_work']), dtype=float)
                    except:
                        continue

                    centroid_value=centroid(to_calculate)
                    cluster_centers.append(centroid_value)
                    medoid_value=medoid(to_calculate)
                    latitude_home = user_home_coords[0]
                    latitude_work = user_work_coords[0]
                    # print(f"{centroid_value}, home {user_home_coords}, work {user_work_coords}\n")
                    # print(f"{medoid_value}, home {user_home_coords}, work {user_work_coords}\n")
                    centroid_home=degrees_to_meters(latitude_home,*(centroid_value-user_home_coords))
                    centroid_work=degrees_to_meters(latitude_work,*(centroid_value-user_work_coords))
                    medoid_home=degrees_to_meters(latitude_home,*(medoid_value-user_home_coords))
                    medoid_work=degrees_to_meters(latitude_work,*(medoid_value-user_work_coords))
                    # print(f"Cluster: {i} user: {u}\n")
                    # print(f"Difference between centroid, home {centroid_home} and work {centroid_work}\n")
                    # print(f"Difference between medoid, home {medoid_home} and work {medoid_work}\n")
                    centroids_home.append(sum(centroid_home))
                    centroids_work.append(sum(centroid_work))
                cluster_centers = np.array(cluster_centers)
                #if len(cluster_centers) < 2:
                    # If there's only one cluster center, skip to avoid errors in building a Voronoi diagram.
                    #continue
                voronoi_labels = assign_voronoi_clusters(user_tweets, cluster_centers)
                user_tweets['voronoi_cluster'] = voronoi_labels

                home_cluster=np.argmin(np.abs(centroids_home))
                home_cluster_dist_value = centroids_home[np.argmin(np.abs(centroids_home))]
                work_cluster=np.argmin(np.abs(centroids_work))
                work_cluster_dist_value = centroids_work[np.argmin(np.abs(centroids_work))]
                user_tweets.loc[user_tweets['cluster']==home_cluster, 'predicted_cluster'] = 'home'
                user_tweets.loc[user_tweets['cluster']==work_cluster, 'predicted_cluster'] = 'work'
                user_tweets.loc[(user_tweets['cluster']!=work_cluster) & (user_tweets['cluster']!=home_cluster), 'predicted_cluster'] = 'outlier'
                home_accuracy,home_precision,home_sensitivity,home_specificity, home_balanced_accuracy,home_F1 = calculate_metrics(user_tweets,"home")
                work_accuracy,work_precision,work_sensitivity,work_specificity, work_balanced_accuracy,work_F1 = calculate_metrics(user_tweets,"work")
                accuracies[0].append(home_accuracy)
                accuracies[1].append(work_accuracy)
                centroid_distances[0].append(home_cluster_dist_value)
                centroid_distances[1].append(work_cluster_dist_value)
                sensitivities[0].append(home_sensitivity)
                sensitivities[1].append(work_sensitivity)
                specificities[0].append(home_specificity)
                specificities[1].append(work_specificity)
                balanced_accuracies[0].append(home_balanced_accuracy)
                balanced_accuracies[1].append(work_balanced_accuracy)
                F1_scores[0].append(home_F1)
                F1_scores[1].append(work_F1)
            if accuracies[0] and accuracies[1]:
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'median_home_accuracy': np.median(accuracies[0]),
                    'mean_home_accuracy': np.mean(accuracies[0]),
                    'median_work_accuracy': np.median(accuracies[1]),
                    'mean_work_accuracy': np.mean(accuracies[1]),
                    'home_centroid_mean_distance':np.mean(centroid_distances[0]),
                    'work_centroid_mean_distance':np.mean(centroid_distances[1]),
                    'mean_home_sensitivity' : np.mean(sensitivities[0]),
                    'median_home_sensitivity' : np.median(sensitivities[0]),
                    'mean_work_sensitivity' : np.mean(sensitivities[1]),
                    'median_work_sensitivity' : np.median(sensitivities[1]),
                    'mean_home_specificity' : np.mean(specificities[0]),
                    'median_home_specificity' : np.median(specificities[0]),
                    'mean_work_specificity' : np.mean(specificities[1]),
                    'median_work_specificity' : np.median(specificities[1]),
                    'mean_home_balanced_accuracy' : np.mean(balanced_accuracies[0]),
                    'median_home_balanced_accuracy' : np.median(balanced_accuracies[0]),
                    'mean_work_balanced_accuracy': np.mean(balanced_accuracies[1]),
                    'median_work_balanced_accuracy':np.median(balanced_accuracies[1]),
                    'mean_home_f1':np.mean(F1_scores[0]),
                    'median_home_f1':np.median(F1_scores[0]),
                    'mean_work_f1':np.mean(F1_scores[1]),
                    'median_work_f1':np.median(F1_scores[1]),
                })
            else:
                break
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='mean_home_accuracy', ascending=False, inplace=True)
    results_df.to_csv(f"results_{dataset_path_without_csv}.csv",index=False)
    print(results_df)
#print(statistics.median(home_accuracies))
#print(statistics.mean(home_accuracies))