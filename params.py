import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = "200k"
CHUNK_SIZE = 200
GCP_PROJECT = "<your project id>" # TO COMPLETE
BQ_DATASET = "..."
BQ_REGION = "EU"
MODEL_TARGET = "local"
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")

COLUMN_NAMES_RAW = ['Type', 'Sub type', 'Sequence', 'Location', 'Property Type', 'Bedrooms',
       'Size (Sqf)', 'Land Size', 'Amount (AED)', 'AED/Sqf', 'Developer',
       'Property Name_property', 'Location_property', 'coordinates_property',
       'name_property', 'types_property', 'address_property',
       'rating_property', 'user_ratings_total_property', 'northeast_property',
       'southwest_property', 'latitude', 'longitude', 'Month_Year']

DTYPES_RAW = {
    "Type": "object",
    "Sub type": "object",
    "Sequence": "object",
    "Location": "object",
    "Property Type": "object",
    "Bedrooms": "int64",
    "Size (Sqf)": "float64",
    "Land Size": "float64",
    "Amount (AED)": "float64",
    "AED/Sqf": "float64",
    "Developer": "object",
    "Property Name_property": "object",
    "Location_property": "object",
    "coordinates_property": "object",
    "name_property": "object",
    "types_property": "object",
    "address_property": "object",
    "rating_property": "float64",
    "user_ratings_total_property": "float64",
    "northeast_property": "object",
    "southwest_property": "object",
    "latitude": "float64",
    "longitude": "float64",
    "Month_Year": "object"
}

DTYPES_PROCESSED = np.float32

