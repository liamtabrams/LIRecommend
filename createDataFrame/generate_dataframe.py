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

#OPTIONAL/UP IN THE AIR FOR THE TIME BEING
'''import numpy as np
# Replace 'N/A' with NaN in the whole DataFrame
df.replace('N/A', np.nan, inplace=True)'''

df.head()