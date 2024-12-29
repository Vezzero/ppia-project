import pandas as pd
import time
import numpy.random as nr
import geopy
import random
import math

# Funzione per calcolare le coordinate a una distanza specifica
def generate_location_near(lat, lon, min_distance_km, max_distance_km):
    # Raggio terrestre approssimativo in km
    earth_radius = 6371.0

    # Genera una distanza casuale tra il minimo e il massimo
    distance_km = random.uniform(min_distance_km, max_distance_km)

    # Genera un angolo casuale (in radianti)
    angle = random.uniform(0, 2 * math.pi)

    # Calcola lo spostamento in gradi di latitudine e longitudine
    delta_lat = (distance_km / earth_radius) * math.cos(angle)
    delta_lon = (distance_km / earth_radius) * math.sin(angle) / math.cos(math.radians(lat))

    # Nuove coordinate
    new_lat = lat + math.degrees(delta_lat)
    new_lon = lon + math.degrees(delta_lon)
    return new_lat, new_lon


class Tweet:
    def __init__(self,id, user, coords, timestamp):
        self.id= id
        self.coords = coords
        self.user = user
        self.timestamp = timestamp

    def __str__(self):
        return f"tweetId:{self.id} {self.coords} {self.user.username} ora:{self.timestamp}"
    
class User:
    def __init__(self, username, home, work="Aprilia"):
        self.username = username
        self.home = home
        self.work = work
    def __str__(self):
        return f"{self.username}\n home:{self.home}\n work:{self.work}"
    

r = nr.default_rng(42)
r1 = r.random()
r2 = r.random()
print(r1)
print(r2)
num_users = 10
post_home_user = 10
post_work_user = 1
users = []
tweets = []
# Generate users, with random home and work locations
for i in range(num_users):
    home_lat = random.uniform(24.396308, 49.384358)
    home_lon = random.uniform(-125.0, -66.93457)
    home = [home_lat,home_lon]
    work_lat, work_lon = generate_location_near(home_lat, home_lon, 10, 20)
    work = [work_lat,work_lon]
    users.append(User(f"u{i}",home,work))

# Count effective tweet ids
tweet_id = 0
#TODO evaluate effective distances 
for u in users:
    # TODO generate variable number of tweets "at home"
    for i in range(post_home_user):
        #noise = r.normal(size=2,loc=0,scale=0.1)
        #noisy_home = u.home + noise
        noisy_home=generate_location_near(u.home[0],u.home[1],0,0.05)
        tweets.append(Tweet(f"t{tweet_id}",u,noisy_home,time.time()))
        tweet_id += 1
    # TODO generate vbariable number of tweets "at work"
    for i in range(post_work_user):
        #noise = r.normal(size=2,loc=0,scale=0.1)
        #noisy_home = u.home + noise
        noisy_work=generate_location_near(u.work[0],u.work[1],0,0.05)
        tweets.append(Tweet(f"t{tweet_id}",u,noisy_work,time.time()))
        tweet_id += 1
    # TODO generate variable number of tweets outliers

user_tweets = []
for t in tweets:
    user_tweets.append(
        {
                "TweetID": f"{t.id}",
                "User": t.user.username,
                "Latitude": t.coords[0],
                "Longitude": t.coords[1],
                "Timestamp": t.timestamp,
                "User_home": t.user.home,
                "User_work":t.user.work
            }
    )


adjusted_tweets_df = pd.DataFrame(user_tweets)

# Salva il dataset su un file CSV
adjusted_file_path = "tweets.csv"
adjusted_tweets_df.to_csv(adjusted_file_path, index=False)

print(f"Dataset salvato in: {adjusted_file_path}")
