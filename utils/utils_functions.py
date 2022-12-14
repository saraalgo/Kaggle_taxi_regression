## UTILS TO USE OVER THE GIT REPOSITORY
## -------------------------------------------------------------------##
import os

# Feature engineering packages
from geopy import distance
import gmaps
import re
import math

##1. Function to create new folder
def folder_create(folder):
    """
    Function to check if a folder exist, otherwise, create one named like indicated
    :params: folder - name of the new folder 
    :return: 
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

##2. Function to get distance feature
def get_distance(row): 
    """
    Function to extract from geopy package the distance from latitude and longitude
    :params: row - each row of the dataset with all the columns, essential to have latitude and longitude 
    :return: distance new column
    """
    pick = (row.pickup_latitude, row.pickup_longitude)
    drop = (row.dropoff_latitude, row.dropoff_longitude)
    dist = distance.geodesic(pick, drop).km
    return dist


##3. Function to convert miles and ft in km
def to_km(distance):
    """
    Function to convert miles/ft to km
    :params: distance - distance with numbers and metric from googlemaps
    :return: km equivalence
    """
    if "mi" in distance:
        miles = float(re.search("\d+\.\d+", distance).group())
        km = miles * 1.60934
    else:
        ft = int(re.search("\d+", distance).group())
        km = ft * 0.0003048
    return km

##4. Function to convert hours in seconds
def to_seconds(duration):
    """
    Function to convert hours/minutes to seconds
    :params: duration - duration with numbers and metric from googlemaps
    :return: seconds equivalence
    """
    if "mins" in duration:
        seconds = int(re.search("\d+", duration).group())*60
    elif "s" in duration:
        seconds = int(re.search("\d+", duration).group())
    else:
        numbers = [int(new_string) for new_string in str.split(duration) if new_string.isdigit()]
        hours = numbers[0]
        minutes = numbers[1]
        seconds = (hours*3600) + (minutes*60)
    return seconds


##5. Function to get distance from local data and distance and duration features from gmaps
def get_maps_features(row): 
    """
    Function to extract from gmaps package the distance and duration of a trip
    :params: row - each row of the dataset with all the columns, essential to have latitude and longitude 
    :return: new row including the two new features
    """
    pick = str(row.pickup_latitude) + "," + str(row.pickup_longitude)
    drop = str(row.dropoff_latitude) + "," + str(row.dropoff_longitude)
    directions = gmaps.directions(pick, drop, mode="driving",
                                  avoid = "ferries")
    row["distance_maps"] = to_km(directions[0]['legs'][0]['distance']['text'])
    row["duration_maps"] = to_seconds(directions[0]['legs'][0]['duration']['text'])
    return row


##6. Revert log2
def revert_log2(x):
    """
    Function to revert log2 and check the final trip duration prediction
    :params: x - np.array of y predictions 
    :return: prediction of duration of the trip in seconds
    """
    return math.pow(2,x)