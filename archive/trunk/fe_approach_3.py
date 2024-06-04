json_files_path = "drive/MyDrive/LI-Jobs-JSON/"
links_csv_path = "drive/MyDrive/successfulLinksGDRIVE.csv"

import pandas as pd
links_dataframe = pd.read_csv(links_csv_path, header=None, names=['url', 'rating'])
#links_dataframe.head()
#print(links_dataframe.loc[0, 'rating'])

# let's snakecase our column names to avoid having spaces in them
# also we add a rating column at the end of the list to store the target variable
df_columns = ["employment_type", "job_function", "description_of_product/service", "industries",
              "position_name", "broader_role_name", "company", #"location", "salary/compensation_range",
              "responsibilities", "goals/objectives", "name_of_department/team", "required_qualifications",
              "preferred_qualifications", "benefits", "work_arrangement", "city", "state", "country",
              "min_salary", "max_salary", "rating"]
# Initialize DataFrame with column names
df = pd.DataFrame(columns=df_columns)

import os
import json
locale_json_files_path = "drive/MyDrive/jobLocations/"
salary_json_files_path = "drive/MyDrive/jobSalaries/"

for i in range(1, 108):
  row_num = df.last_valid_index()
  print(row_num)
  if row_num == None:
    row_num = 0
  else:
    row_num = row_num + 1
  assert i == row_num + 1
  json_file_path = os.path.join(json_files_path, f'row{i}.json')
  # Read JSON file
  with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)
  if 'fields' in json_data.keys():
    #print(json_data)
    field_names = json_data['fields']
    for index, value in enumerate(field_names):
      info = json_data['info'][index]
      # Check if the value exists as a column name (ignoring case)
      column_name = value.replace(" ", "_").lower()
      if column_name in df.columns:
        # Add the info to the corresponding column and row
        df.at[row_num, column_name] = info
  else:
    for key, value in json_data.items():
      # Check if the key exists as a column name (ignoring case)
      column_name = key.replace(" ", "_").lower()
      if column_name == 'emploment_type':
        column_name = 'employment_type'
      if column_name in df.columns:
        # Add the value to the corresponding column and row
        df.at[row_num, column_name] = value

  locale_json_file_path = os.path.join(locale_json_files_path, f'row{i}.json')
  salary_json_file_path = os.path.join(salary_json_files_path, f'row{i}.json')
  # Read locale JSON file
  with open(locale_json_file_path, 'r') as json_file:
    json_data = json.load(json_file)
  try:
    city = json_data["city"]
    state = json_data["state"]
    country = json_data["country"]
    df.at[row_num, 'city'] = city
    df.at[row_num, 'state'] = state
    df.at[row_num, 'country'] = country
  except:
    df.at[row_num, 'city'] = "N/A"
    df.at[row_num, 'state'] = "N/A"
    df.at[row_num, 'country'] = "N/A"
  # Read salary JSON file
  with open(salary_json_file_path, 'r') as json_file:
    json_data = json.load(json_file)
  min_salary = json_data["salary_min"]
  max_salary = json_data["salary_max"]
  df.at[row_num, 'min_salary'] = min_salary
  df.at[row_num, 'max_salary'] = max_salary
  rating = links_dataframe.loc[row_num, 'rating']
  df.at[row_num, 'rating'] = rating

import numpy as np
# Replace 'N/A' with NaN in the whole DataFrame
df.replace('N/A', np.nan, inplace=True)

links_csv_path = "drive/MyDrive/successfulLinksGDRIVE.csv"

import pandas as pd
links_dataframe = pd.read_csv(links_csv_path, header=None, names=['url', 'rating'])
#links_dataframe.head()
#print(links_dataframe.loc[0, 'rating'])

# let's snakecase our column names to avoid having spaces in them
# also we add a rating column at the end of the list to store the target variable
df_columns = ["posting_text", "rating"]
# Initialize DataFrame with column names
df2 = pd.DataFrame(columns=df_columns)

for i in range(1, 108):
  row_num = df2.last_valid_index()
  print(row_num)
  if row_num == None:
    row_num = 0
  else:
    row_num = row_num + 1
  assert i == row_num + 1
  text_file_path = f'drive/MyDrive/text_files2/row{i}.txt'
  with open(text_file_path, 'r') as file:
    # Read the entire content of the file into a string
    file_contents = file.read()
  df2.at[row_num, 'posting_text'] = file_contents
  rating = links_dataframe.loc[row_num, 'rating']
  df2.at[row_num, 'rating'] = rating

import numpy as np
# Replace 'N/A' with NaN in the whole DataFrame
df2.replace('N/A', np.nan, inplace=True)

salary_df = df[['min_salary', 'max_salary']]
# Iterate over columns and fill NaN values with the mode of each column
for col in salary_df.columns:
  mode_val = salary_df[col].mode()[0]
  salary_df[col].fillna(mode_val, inplace=True)  # Fill NaN values with the mode

# Concatenate the DataFrames
concatenated_df = pd.concat([df2, salary_df], axis=1)

print(concatenated_df.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Assuming your dataframe is named 'df' and the text column is named 'job_posting_text'

# Step 1: Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(concatenated_df[['posting_text', 'min_salary', 'max_salary']], concatenated_df['rating'], test_size=0.2, random_state=507)

# Step 2: Initialize and fit TF-IDF vectorizer on the training data only
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['posting_text'])
# Convert TF-IDF matrix to DataFrame
X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=X_train.index)

# Step 3: Transform the test data using the fitted vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test['posting_text'])
# Convert TF-IDF matrix to DataFrame
X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=X_test.index)

new_df = X_train.drop(columns='posting_text')
new_test_df = X_test.drop(columns='posting_text')
X_train = pd.concat([new_df, X_train_tfidf_df], axis=1)
X_test = pd.concat([new_test_df, X_test_tfidf_df], axis=1)


# Now you have X_train and X_test ready to be used with your model for training and evaluation.