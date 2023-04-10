from datetime import datetime
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Read data
df = pd.read_csv('data/merged_cleaned_data.csv')

# Get 200k rows
df['Month_Year'] = pd.to_datetime(df['Month_Year'])
df = df.sort_values(by='Month_Year', ascending=False)
df = df.iloc[:200000]

# Pre-Processing
def extract_year(date_str):
    date_obj = datetime.strptime(date_str, '%m-%Y')
    return date_obj.year

stop_words = set(stopwords.words('english'))

df['Month_Year'] = df['Month_Year'].apply(lambda x: datetime.strftime(x, '%m-%Y'))
df['Year'] = df['Month_Year'].apply(extract_year)

df['Property Name'] = df['Property Name'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
df['Property Name'] = df['Property Name'].str.lower()
df['Property Name'] = df['Property Name'].str.replace('[^\w\s]', '')
df['Property Name'] = df['Property Name'].apply(lambda x: word_tokenize(x))

# Hashing 'name_property'
vectorizer = HashingVectorizer(n_features=200000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['Property Name'].astype(str))
svd = TruncatedSVD(n_components=100)
components = svd.fit_transform(X)

hashing_columns = []
for i in range(1, (components.shape[1]+1)):
    hashing_columns.append('hashing_'+str(i))

components_df = pd.DataFrame(components, columns=hashing_columns)
df = pd.concat([df, components_df], axis=1)

# Standardization
cols_to_scale = ['latitude', 'Land Size','Year','Bedrooms','Size (Sqf)','longitude']
scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Encoding
dummy_cols = ['Sequence', 'Type', 'Sub type', 'Property Type','Location']
df = pd.get_dummies(df, columns=dummy_cols)
df.drop(['AED/Sqf', 'Developer', 'name_property_google', 'types_google',
         'address_google', 'rating_google', 'user_ratings_total_google',
         'northeast_google', 'southwest_google', 'Month_Year', 'Property Name'], axis=1, inplace=True)

# Training
X = df.drop(['Amount (AED)'], axis=1)
y = df['Amount (AED)']

param_grid = {
    'n_estimators': 200,
    'max_depth': 17,
    'learning_rate': 0.08837483,
    'max_delta_step':0,
    'alpha': 0.41610393, 
    'lambda':0.58841556,
    'gamma': 0.2964622,
    'min_child_weight':0.52404153,
    'subsample': 0.8800853,
    'colsample_bytree': 0.6815701,
}


model = XGBRegressor(**param_grid)
model.fit(X, y)

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()

    # Preprocess the data
    property_name = data['property_name']
    # ... preprocessing code ...

    # Hashing 'name_property'
    X = vectorizer.transform([property_name])
    components = svd.transform(X)
    components_df = pd.DataFrame(components, columns=hashing_columns)

    # Standardization
    cols_to_scale = ['latitude', 'Land Size','Year','Bedrooms','Size (Sqf)','longitude']
    input_data = pd.DataFrame(data, columns=cols_to_scale)
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # Encoding
    input_data = pd.get_dummies(input_data, columns=dummy_cols)

    # Combine input data with hashing components
    input_data = pd.concat([input_data, components_df], axis=1)

    # Make prediction
    prediction = model.predict(input_data)

    # Return the result
    return jsonify({'prediction': prediction[0]})
