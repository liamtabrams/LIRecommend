from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
#from fastapi.templating import Jinja2Templates as templates
#from fastapi.responses import HTMLResponse
from openai import OpenAI
import os
import json

def scrape_from_link(url):
  dir_path = "app/user_input/scraped_text"
  posting_ind = len(os.listdir(dir_path))
  text_file_path = f"app/user_input/scraped_text/posting{posting_ind}.txt"
  with open(text_file_path, 'w') as file:
    resp = requests.get(fr"{url}")
    if resp.status_code == 200:
      # get the response text. in this case it is HTML
      html = resp.text
      # Parse the HTML content
      soup = BeautifulSoup(html, 'html.parser')

    else:
      time.sleep(1)
      # request again
      resp = requests.get(fr"{url}")
      if resp.status_code == 200:
        # get the response text. in this case it is HTML
        html = resp.text
        # Parse the HTML content
        soup = BeautifulSoup(html, 'html.parser')

      else:
        print("Failed to retrieve job posting from", fr"{url}")

    # get position, company, location, pay
    x = soup.get_text().split('\n')
    # Remove elements with only whitespace
    filtered_list = [string for string in x if string.strip()]

    for i in filtered_list:
      if i.find('Join now') != -1 and filtered_list[filtered_list.index(i) + 1].find('Sign in') != -1:
        position = filtered_list[filtered_list.index(i) + 2]
        company = filtered_list[filtered_list.index(i) + 3]
        location = filtered_list[filtered_list.index(i) + 4]
        break

    for i in filtered_list:
      if i.find('Base pay range') != -1:
        salary_index = filtered_list.index(i) + 1
        salary = filtered_list[salary_index]
        salary = salary.lstrip()
        salary = "N/A" if (salary.find("$") == salary.find("€") == salary.find("£") == -1) else salary
        break
      else:
        salary = "N/A"

    file.write(f"position is {position}\n")
    file.write(f"company is {company}\n")
    file.write(f"location is {location}\n")
    file.write(f"salary is {salary}\n\n")

    #get seniority level, employment type, job function, and industries
    spans = soup.find_all('span', {'class': "description__job-criteria-text description__job-criteria-text--criteria"})
    for span in spans:
      parent_tags = span.parent.find_all("h3", {'class': "description__job-criteria-subheader"})
      for tag in parent_tags:
        field = tag.contents[0].strip()
        #print(span.parent.find_all("h3", {'class': "description__job-criteria-subheader"}))
      value = span.contents[0].strip()
      file.write(f"{field} is {value}\n")
    file.write("\n")

    # get main body text
    characters_per_line = []

    # Extract text content from the HTML
    text_content = soup.get_text()

    # Split the text into lines
    lines = text_content.splitlines()

    # Calculate the number of characters in each line
    for line in lines:
      characters_per_line.append(len(line))

    body_len = max(characters_per_line)
    body_idx = characters_per_line.index(body_len)
    body_text = lines[body_idx]
    file.write(body_text)
  return text_file_path, posting_ind

def generate_salary_json_file(posting_ind, salary_val):
  prompt = "Take the following python data and infer the minimum and maximum of the salary range, and fill out their values as floating point numbers with three decimal places in units of thousands in a JSON dictionary, with 'salary_min' and 'salary_max' being the keys. If the data given is ['$150,000.00/yr - $220,000.00/yr'] then you should return {'salary_min': 150, 'salary_max': 220}, but make sure to use double quotes to enclose the key names. If you infer that info is in dollars per hour, convert the numbers to annual salary in thousands so output is same regardless of given units. Note that $48/hr is equal to $100,000/yr. Put 'N/A' under the fields if the required information is not given. If only one number is given put it under 'salary_max'. Return only the JSON dictionary. I want you to do it, not to tell me how to code it. I want you to do it for: "

  prompt = prompt + "/n" + str(salary_val)

  client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-proj-2Iut0hiaVnKHtMPvhKcqT3BlbkFJ4bGC2VbEkNT1SN2eYghm",
  )

  chat_completion = client.chat.completions.create(
      messages=[
          {
              "role": "user",
              "content": prompt,
          }
      ],
      model="gpt-3.5-turbo",
  )
  json_files_dir = "app/user_input/salariesJSON/"
  json_data = json.loads(chat_completion.choices[0].message.content.strip("`").strip('json').strip())
  json_file_path = json_files_dir + f'posting{posting_ind}.json'
  # Write JSON data to the file
  with open(json_file_path, "w") as json_file:
    json.dump(json_data, json_file)

  return json_file_path

def generate_dataset_input(url):
  text_file_path, posting_ind = scrape_from_link(url)
  print(os.listdir("app/user_input/scraped_text"))
  with open(text_file_path, "r") as f:
    file_contents = f.read()
    f.seek(0)
    lines = f.readlines()

  datapoint_dict = {}
  salary_line = lines[3]
  salary_val = salary_line.strip("\n").strip("salary is ")
  json_file_path = generate_salary_json_file(posting_ind, salary_val)
  datapoint_dict['posting_text'] = file_contents
  with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)
  min_salary = json_data["salary_min"]
  max_salary = json_data["salary_max"]
  dataset_df = pd.read_csv('app/dataset/myDataset.csv')
  min_salary_mode = dataset_df['min_salary'].mode()[0]
  max_salary_mode = dataset_df['max_salary'].mode()[0]
  if min_salary == "N/A":
    min_salary = min_salary_mode
  if max_salary == "N/A":
    max_salary = max_salary_mode  
  datapoint_dict['min_salary'] = min_salary
  datapoint_dict['max_salary'] = max_salary

  return datapoint_dict

app = FastAPI()

# Mount the static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
def read_root():
    return {'message': "Liam's Job Preference Model API"}


def scp_file_from_container(source_path, destination_path):
    # Construct the scp command
    scp_command = [
        "scp",
        f"{source_path}",
        destination_path
    ]

    # Execute the scp command
    subprocess.run(scp_command)

# Landing page endpoint
'''@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("landing_page.html", {"request": request})'''

# Predict Salary page
'''@app.get("/predict-rating", response_class=HTMLResponse)
async def predict_rating(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Collect Data page
@app.get("/collect-data", response_class=HTMLResponse)
async def collect_data(request: Request):
    return templates.TemplateResponse("collect_data.html", {"request": request})'''

@app.post('/predict')
def predict(data: dict):
    features = np.array(data['features']).reshape(1, -1).flatten().tolist()
    datapoint = generate_dataset_input(features[0])
    model = joblib.load('app/models/linreg_clf.joblib')
    tfidf_vect = joblib.load('app/models/tfidf_vectorizer.joblib')
    # Step 3: Transform the test data using the fitted vectorizer
    X_test_tfidf = tfidf_vect.transform([datapoint['posting_text']])
    # Convert TF-IDF matrix to DataFrame
    X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vect.get_feature_names_out())
    # Create DataFrame excluding 'posting_text' column
    new_df = pd.DataFrame({key: [value] for key, value in datapoint.items() if key != 'posting_text'})
    X_test = pd.concat([new_df, X_test_tfidf_df], axis=1)
    prediction = model.predict(X_test)
    return {'prediction': prediction[0]}
    

@app.post('/submit-data')
def append_datapoint(data: dict):
    datapoint_dict = generate_dataset_input(data['url'])
    datapoint_dict['rating'] = int(data['rating'])

    # Create a new DataFrame with the new row
    new_row_df = pd.DataFrame([datapoint_dict])

    dataset_df = pd.read_csv('app/dataset/myDataset.csv')
    
    df = pd.concat([dataset_df, new_row_df], ignore_index=True)

    df.to_csv('app/dataset/myDataset.csv', index=False)

    updated_df = pd.read_csv('app/dataset/myDataset.csv')
    last_row_dict = updated_df.iloc[-1].to_dict()

    # Assert that the last row of the DataFrame has all the same data as the dictionary object
    assert datapoint_dict == last_row_dict
  
    print(updated_df.tail())


@app.post('/retrain-model')
def retrain_model():
    train_linreg('app/dataset/myDataset.csv')

'''in this code we will use the entire dataset as a training set for the linear
regression model'''

def train_linreg(dataset_path):
  #from sklearn.model_selection import train_test_split
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.model_selection import cross_val_score
  from sklearn.metrics import make_scorer, accuracy_score
  import joblib
  import pandas as pd
  from sklearn.linear_model import LinearRegression
  from sklearn.preprocessing import LabelEncoder

  # Define your custom scoring function
  def custom_scoring_function(y_true, y_pred):
    # Define your custom scoring logic
    # For example, let's say you want to calculate the mean absolute error (MAE)
    y_pred = np.clip(y_pred, 0, 3)
    y_pred = [round(pred) for pred in y_pred]
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy

  # Make a scorer from the custom scoring function
  custom_scorer = make_scorer(custom_scoring_function, greater_is_better=True)
  # Helper function to display key metrics for model performance.
  def display_scores(scores):
    print('Accuracy of rounded and clipped predictions: results of 10-fold cross val')
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

  dataset = pd.read_csv(dataset_path)

  # Step 1: Split data into training and testing sets
  #X_train, X_test, y_train, y_test = train_test_split(dataset[['posting_text', 'min_salary', 'max_salary']], dataset['rating'], test_size=0.2, random_state=42)

  X = dataset[['posting_text', 'min_salary', 'max_salary']]
  y = dataset['rating']
  le = LabelEncoder()
  y_train = le.fit_transform(dataset['rating'])

  # Step 2: Initialize and fit TF-IDF vectorizer on the training data only
  tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
  X_train_tfidf = tfidf_vectorizer.fit_transform(X['posting_text'])
  # Convert TF-IDF matrix to DataFrame
  X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=X.index)

  # Step 3: Transform the test data using the fitted vectorizer
  #X_test_tfidf = tfidf_vectorizer.transform(X_test['posting_text'])
  # Convert TF-IDF matrix to DataFrame
  #X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=X_test.index)

  new_df = X.drop(columns='posting_text')
  X_train = pd.concat([new_df, X_train_tfidf_df], axis=1)

  # Now X_train and X_test are ready to be passed to a model
  model = LinearRegression()
  model.fit(X_train, y_train)
  model_scores = cross_val_score(model, X_train, y_train, scoring=custom_scorer, cv = 10)
  display_scores(model_scores)

  # Save the TFIDF model to a file
  joblib.dump(tfidf_vectorizer, 'app/models/tfidf_vectorizer.joblib')

  # Serialise model and dump on disk
  joblib.dump(model, 'app/models/linreg_clf.joblib')