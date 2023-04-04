

from datetime import datetime
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD


# # Read data

# In[2]:


df = pd.read_csv('data/merged_cleaned_data.csv')


# In[3]:


# Get 200k rows
df['Month_Year'] = pd.to_datetime(df['Month_Year'])
df = df.sort_values(by='Month_Year', ascending=False)

# select the most recent 200k rows using iloc
df = df.iloc[:200000]


# # Pre-Processing

# Create Year column

# In[4]:


df['Month_Year'] = df['Month_Year'].apply(lambda x: datetime.strftime(x, '%m-%Y'))
# define a function to convert the string to a datetime object and extract the year
def extract_year(date_str):
    date_obj = datetime.strptime(date_str, '%m-%Y')
    return date_obj.year

# apply the function to the date column and create a new column with only the year
df['Year'] = df['Month_Year'].apply(extract_year)


# Tokenize 'name_property'

# In[6]:


# Remove stopwords
stop_words = set(stopwords.words('english'))
df['Property Name'] = df['Property Name'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Convert to lowercase
df['Property Name'] = df['Property Name'].str.lower()

# Remove punctuation and special characters
df['Property Name'] = df['Property Name'].str.replace('[^\w\s]', '')

# Tokenize the text
df['Property Name'] = df['Property Name'].apply(lambda x: word_tokenize(x))

df.shape


# Hashing 'name_property'

# In[147]:


vectorizer = HashingVectorizer(n_features=200000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['Property Name'].astype(str))
svd = TruncatedSVD(n_components=100)
components = svd.fit_transform(X)

hashing_columns = []
# loop over the range from 1 to 100 and create a string for each number
for i in range(1, (components.shape[1]+1)):
    hashing_columns.append('hashing_'+str(i))

# create a new DataFrame with the SVD components
components_df = pd.DataFrame(components, columns=hashing_columns)

# merge the original DataFrame with the components DataFrame
df = pd.concat([df, components_df], axis=1)


# Standardization

# In[149]:


from sklearn.preprocessing import StandardScaler

# create an instance of the StandardScaler class
scaler = StandardScaler()

# select the columns to scale
cols_to_scale = ['latitude', 'Land Size','Year','Bedrooms','Size (Sqf)','longitude']

# apply the scaler to the selected columns
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


# Encoding

# In[151]:


dummy_cols = ['Sequence', 'Type', 'Sub type', 'Property Type','Location']
df = pd.get_dummies(df, columns=dummy_cols)

# print the resulting DataFrame


# In[155]:


df.drop(['AED/Sqf', 'Developer', 'name_property_google', 'types_google',
       'address_google', 'rating_google', 'user_ratings_total_google',
       'northeast_google', 'southwest_google', 'Month_Year', 'Property Name'], axis=1, inplace=True)


# # Training

# Train-test split

# In[164]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Define the features and target variable
X = df.drop(['Amount (AED)'], axis=1)  # Assuming 'Price' is your target variable
y = df['Amount (AED)']


# Model

# In[175]:


from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

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
    #'scale_pos_weight':1,
}

model = XGBRegressor(**param_grid)
#random_search = RandomizedSearchCV(model, param_distributions=param_grid)
model.fit(X, y)


# Evaluation
