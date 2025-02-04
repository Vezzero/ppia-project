import pandas as pd
import time
import numpy.random as nr
import numpy as np
import random
import math
from datetime import datetime, timedelta

r = nr.default_rng(42)
# Funzione per calcolare le coordinate a una distanza specifica
def generate_location_near(lat, lon, min_distance_km, max_distance_km):
    # Raggio terrestre approssimativo in km
    earth_radius = 6371.0

    # Genera una distanza casuale tra il minimo e il massimo
    distance_km = r.uniform(min_distance_km, max_distance_km)

    # Genera un angolo casuale (in radianti)
    angle = r.uniform(0, 2 * math.pi)

    # Calcola lo spostamento in gradi di latitudine e longitudine
    delta_lat = (distance_km / earth_radius) * math.cos(angle)
    delta_lon = (distance_km / earth_radius) * math.sin(angle) / math.cos(math.radians(lat))

    # Nuove coordinate
    new_lat = lat + math.degrees(delta_lat)
    new_lon = lon + math.degrees(delta_lon)
    return new_lat, new_lon

class Tweet:
    def __init__(self, id, user, coords, timestamp, cluster):
        self.id = id
        self.coords = coords
        self.user = user
        self.timestamp = timestamp
        self.cluster = cluster
    def __str__(self):
        return f"tweetId:{self.id} {self.coords} {self.user.username} ora:{self.timestamp}"

class User:
    def __init__(self, username, home, work):
        self.username = username
        self.home = home
        self.work = work

    def __str__(self):
        return f"{self.username}\n home:{self.home}\n work:{self.work}"

# Function to generate random timestamps within a range
def random_timestamp_in_time_range(start_hour, end_hour):
    current_date = datetime.now().date()

    if start_hour <= end_hour:
        start_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=start_hour)
        end_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=end_hour)
    else:
        if r.choice([True, False]):
            start_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=start_hour)
            end_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=24)
        else:
            start_time = datetime.combine(current_date + timedelta(days=1), datetime.min.time())
            end_time = datetime.combine(current_date + timedelta(days=1), datetime.min.time()) + timedelta(hours=end_hour)

    random_time = start_time + timedelta(seconds=r.integers(0, int((end_time - start_time).total_seconds()),endpoint=True,dtype=int))
    return random_time.timestamp()

r1 = r.random()
r2 = r.random()
print(r1)
print(r2)
num_users = 100
post_home_user = 100
post_work_user = 100
post_outliers = 20
users = []
tweets = []
# Generate users with random home and work locations
for i in range(num_users):
    home_lat = r.uniform(37.7081, 37.8324)
    home_lon = r.uniform(-123.0137, -122.3570)
    home = [home_lat, home_lon]
    #work_lat, work_lon = generate_location_near(home_lat, home_lon, 10, 20)
    work_lat = r.uniform(37.7081, 37.8324)
    work_lon = r.uniform(-123.0137, -122.3570)
    work = [work_lat, work_lon]
    users.append(User(f"u{i}", home, work))

# Count effective tweet ids
tweet_id = 0
cluster_id = 0
for u in users:
    # Generate tweets "at home" (8 PM to 7 AM)
    for _ in range(post_home_user):
        # noise = r.normal(size=2,loc=0,scale=0.005)
        # noisy_home = u.home + noise
        noisy_home = generate_location_near(u.home[0], u.home[1], 0, 0.2)
        timestamp = random_timestamp_in_time_range(20, 7)
        tweets.append(Tweet(f"t{tweet_id}", u, noisy_home, timestamp, "home"))
        tweet_id += 1
    for _ in range(post_work_user):
        # noise = r.normal(size=2,loc=0,scale=0.005)
        # noisy_work = u.work + noise
        noisy_work = generate_location_near(u.work[0], u.work[1], 0, 0.2)
        timestamp = random_timestamp_in_time_range(9, 17)
        tweets.append(Tweet(f"t{tweet_id}", u, noisy_work, timestamp, "work"))
        tweet_id += 1
    for _ in range(post_outliers):
        #noise = r.normal(size=2,loc=0,scale=0.1)
        #noisy_home = u.home + noise
        outlier_lat = r.uniform(37.7081, 37.8324)
        outlier_lon = r.uniform(-123.0137, -122.3570)
        timestamp = random_timestamp_in_time_range(0, 23)
        tweets.append(Tweet(f"t{tweet_id}", u, [outlier_lat, outlier_lon], timestamp, "outlier"))
        tweet_id += 1

user_tweets = []
for t in tweets:
    user_tweets.append(
        {
            "TweetID": t.id,
            "User": t.user.username,
            "Latitude": t.coords[0],
            "Longitude": t.coords[1],
            "Timestamp": datetime.fromtimestamp(t.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "User_home": t.user.home,
            "User_work": t.user.work,
            "cluster_id": t.cluster
        }
    )

adjusted_tweets_df = pd.DataFrame(user_tweets)

# Salva il dataset su un file CSV
adjusted_file_path = "tweets.csv"
adjusted_tweets_df.to_csv(adjusted_file_path, index=False)

print(f"Dataset salvato in: {adjusted_file_path}")