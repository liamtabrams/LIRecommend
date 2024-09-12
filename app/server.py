import os
import time
import json
import requests
import logging
import re
import numpy as np
import pandas as pd
import joblib
from bs4 import BeautifulSoup
import multiprocessing
from multiprocessing import Process, shared_memory
import io
import zipfile
import csv
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates as templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import parallel_backend
import math
import matplotlib.pyplot as plt
from crewai import Agent, Task, Crew
import pdb
import time


# Configure the logging
logging.basicConfig(
    level=logging.DEBUG,  # Capture all log levels
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console (stdout)
        logging.FileHandler("app/logs/app.log")  # Logs to a file
    ]
)

logger = logging.getLogger(__name__)

def scrape_from_link(url):
  logger.info(f"Received URL {url} from user")
  
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
      logger.info(f"Successfully parsed {url}")

    else:
      logger.debug(f"{resp.status_code} error code occurred scraping from {url}")
      time.sleep(1)
      # request again
      resp = requests.get(fr"{url}")
      if resp.status_code == 200:
        # get the response text. in this case it is HTML
        html = resp.text
        # Parse the HTML content
        soup = BeautifulSoup(html, 'html.parser')
        logger.info(f"Successfully parsed {url}")

      else:
        logger.debug(f"{resp.status_code} error code occurred 2nd attempt at scraping from {url}")
        print("Failed to retrieve job posting from", fr"{url}")

    # get position, company, location, pay
    x = soup.get_text().split('\n')
    # Remove elements with only whitespace
    filtered_list = [string for string in x if string.strip()]

    print("Extracting posting metadata")
    logger.info(f"Extracting posting metadata from {url}")
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
    logger.info(f"position is {position}\n")
    logger.info(f"company is {company}\n")
    logger.info(f"location is {location}\n")
    logger.info(f"salary is {salary}\n\n")

    #get seniority level, employment type, job function, and industries
    spans = soup.find_all('span', {'class': "description__job-criteria-text description__job-criteria-text--criteria"})
    for span in spans:
      parent_tags = span.parent.find_all("h3", {'class': "description__job-criteria-subheader"})
      for tag in parent_tags:
        field = tag.contents[0].strip()
        #print(span.parent.find_all("h3", {'class': "description__job-criteria-subheader"}))
      value = span.contents[0].strip()
      file.write(f"{field} is {value}\n")
      logger.info(f"{field} is {value}\n")
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

    logger.info("Extracting posting body")
    
    body_len = max(characters_per_line)
    body_idx = characters_per_line.index(body_len)
    body_text = lines[body_idx]
    file.write(body_text)
  logger.info(f"done scraping text from {url}")
  return text_file_path, posting_ind

# Function to read the API key from a text file
def read_api_key(file_path):
  with open(file_path, 'r') as file:
    api_key = file.read().strip()
  return api_key

# Read the API key from the text file
api_key = read_api_key('app/openai_key.txt')

def generate_salary_json_file(posting_ind, posting_text):
  prompt = "Take the following job posting and infer the minimum and maximum of the salary range, and fill out their values as floating point numbers with three decimal places in units of thousands in a JSON dictionary, with 'salary_min' and 'salary_max' being the keys. If the posting says 'The compensation range for this position is between $150,000.00/yr and $220,000.00/yr' then you should return {'salary_min': 150, 'salary_max': 220}, but make sure to use double quotes to enclose the key names. If you infer that info is in dollars per hour, convert the numbers to annual salary in thousands so output is same regardless of given units. Note that $48/hr is equal to $100,000/yr. Put 'N/A' under the fields if the required information is not given. If only one number is given put it under 'salary_max'. Return only the JSON dictionary. I want you to do it, not to tell me how to code it. I want you to do it for: "

  prompt = prompt + "/n" + posting_text

  client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
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
  try:
    text_file_path, posting_ind = scrape_from_link(url)
  except Exception as e:
    logger.error(f"Encountered error {e} when trying to scrape from {url}")
    return
  
  with open(text_file_path, "r") as f:
    file_contents = f.read()
    f.seek(0)

  datapoint_dict = {}
  logger.info(f"attempting to generate JSON salary file for {url} using ChatGPT API")
  try:
    json_file_path = generate_salary_json_file(posting_ind, file_contents)
    logger.info(f"Successfully created JSON salary file at {json_file_path} relative to the working directory of the container")
  except Exception as e:
    logger.error(f"Encountered error {e} when trying to generate JSON salary file")
    return

  datapoint_dict['posting_text'] = file_contents
  with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)
  min_salary = json_data["salary_min"]
  max_salary = json_data["salary_max"]
  dataset_df = pd.read_csv('app/user_data/dataset/myDataset.csv')
  logger.info("reading in dataset to determine salary column modes")
  min_salary_mode = dataset_df['min_salary'].mode()[0]
  max_salary_mode = dataset_df['max_salary'].mode()[0]
  if min_salary == "N/A":
    min_salary = min_salary_mode
  if max_salary == "N/A":
    max_salary = max_salary_mode  
  datapoint_dict['min_salary'] = min_salary
  datapoint_dict['max_salary'] = max_salary

  logger.info(f"successfully generated dataset input from {url}")
  return datapoint_dict

app = FastAPI()

# Mount the static files
app.mount("/static", StaticFiles(directory="static"), name="static")
logger.info("mounted the '/static' directory to StaticFiles in FastAPI app")

@app.get('/')
def read_root():
    logger.info("Root endpoint called")
    return {'message': "Liam's Job Preference Model API"}

# Landing page endpoint
'''@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("landing_page.html", {"request": request})'''

# Predict Salary page
'''@app.get("/predict-rating", response_class=HTMLResponse)
async def predict_rating(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})'''

# Collect Data page
@app.get("/collect-data", response_class=HTMLResponse)
async def collect_data(request: Request):
    return templates.TemplateResponse("collect_data.html", {"request": request})

def get_rating_color(rating):
    """
    Returns a hex color code for a given rating between 0 and 3.
    
    0 corresponds to red, 3 corresponds to green.
    Intermediate values are interpolated between these colors.
    
    Parameters:
        rating (float): A rating between 0 and 3.
        
    Returns:
        str: A hex color code.
    """
    # Clamp the rating to ensure it's within the expected range
    rating = max(0, min(3, rating))
    
    # Calculate the green component (0 for rating 0, 255 for rating 3)
    green = int((rating / 3) * 120) + 49

    blue = int((rating / 3) * 19) + 49
    
    # Calculate the red component (255 for rating 0, 0 for rating 3)
    red = 255 - green - 2*int(math.sqrt(blue))
    
    # Return the color as a hex code
    return f'#{red:02x}{green:02x}{blue:02x}'

def generate_prediction(url):
    while True:
        try:
            datapoint = generate_dataset_input(url)
            logger.info("generated datapoint")
            model = joblib.load('app/user_data/models/rf_clf.joblib')
            tfidf_vect = joblib.load('app/user_data/models/tfidf_vectorizer.joblib')
            logger.info("loaded pre-trained Linear Regression and TFIDF Vectorizer models")
            # Step 3: Transform the test data using the fitted vectorizer
            X_test_tfidf = tfidf_vect.transform([datapoint['posting_text']])
            logger.info("transformed datapoint's posting_text into TFIDF matrix")
            # Convert TF-IDF matrix to DataFrame
            X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vect.get_feature_names_out())
            logger.info("transformed the matrix into a dataframe")
            #Create DataFrame excluding 'posting_text' column
            new_df = pd.DataFrame({key: [value] for key, value in datapoint.items() if key not in ['posting_text', 'req_quals']})
            X_test = pd.concat([new_df, X_test_tfidf_df], axis=1)
            logger.info("concatenated original columns minus posting_text with TFIDF dataframe")
            prediction = model.predict(X_test)
            logger.info(f"generated prediction {prediction} with the following datapoint: {X_test}")
            color = get_rating_color(prediction)
            logger.info("generated prediction and color code")
            return int(prediction[0]), color, datapoint
        except Exception as e:
          logger.debug(f"ran into {e} when trying to generate prediction")
          print(e)
          pass

def generate_prediction_wrapper(args):
    url = args
    return url, generate_prediction(url)

@app.post('/predict')
def predict(data: dict):
    logger.info("Predict endpoint called")
    url = data["url"]
    logger.info(f"{url} successfully sent to backend")
    prediction, color, datapoint = generate_prediction(url)
    return {'prediction': prediction, 'color': color}

def extract_linkedin_job_urls(html_source):
    # Define the regex pattern to match LinkedIn job URLs
    pattern = r'href="(https://www\.linkedin\.com/jobs/view/[^\s"]+)"'
    
    # Find all matches in the HTML source
    matches = re.findall(pattern, html_source)
    
    # Use a set to remove duplicates
    unique_urls = set(matches)
    
    # Convert the set back to a list (if needed)
    unique_urls_list = list(unique_urls)
    
    return unique_urls_list

@app.post('/recommend')
def recommend(data: dict):
    logger.info("recommend endpoint called")
    term = data.get("term", "")
    if not term:
        return JSONResponse(content={"error": "Job search term is required"}, status_code=400)
        logger.error("No search term specified before calling Recommend")
    
    def convert_to_url_format(term):
        return term.lower().replace(" ", "-")
    
    # Scrape LinkedIn for job postings based on the search term
    jobs_prefix = convert_to_url_format(term)
    search_url = f"https://www.linkedin.com/jobs/{jobs_prefix}-jobs?position=1&pageNum=0"
    print(search_url)
    #headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url)
    html = response.text
    logger.info("Obtained recommendation page source")

    # Parse the job postings from the HTML
    jobs = extract_linkedin_job_urls(html)

    logger.info(f"Obtained the following list of job posting URLs: {jobs}")
    print(jobs)
    
    # Predict ratings for each job posting using multiprocessing
    recommendations = []
    with multiprocessing.Pool() as pool:
        results = pool.map(generate_prediction_wrapper, jobs)
    
    # Collect results
    for url, result in results:
        if result is not None:
            job = {}
            job["url"] = url
            job["rating"] = int(result[0])
            position_line = result[2]["posting_text"].split('\n')[0]
            company_line = result[2]["posting_text"].split('\n')[1]
            position = position_line.strip("position is ")
            company = company_line.strip("company is ")
            job["position"] = position
            job["company"] = company
            job["color"] = result[1]

            # Check if the recommendation matches any row in the blacklist CSV
            with open("app/user_data/blacklist/blacklist.csv", mode="r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if (position == row["position_name"] and 
                        company == row["company"]):
                        break  # Skip this recommendation if it's blacklisted
                else:
                    recommendations.append(job)

    # Sort the job postings by rating (highest first)
    recommendations.sort(key=lambda x: x["rating"], reverse=True)

    # Return the top 10 job postings
    return JSONResponse(content={"recommendations": recommendations[:25]})

@app.post("/blacklist/")
async def add_to_blacklist(data: dict):
    logger.info("blacklist endpoint called")
    position_name = data.get("position_name")
    company = data.get("company")
    
    # Check if any of the required fields are missing
    if not (position_name and company):
        logger.debug(f"Required fields missing in {data}")
        return {"message": "Invalid request. Required fields missing."}
    
    # Add the data to the blacklist.csv file
    with open("app/user_data/blacklist/blacklist.csv", mode="a", newline="") as file:
        fieldnames = ["position_name", "company"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write headers if the file is empty
        if file.tell() == 0:
            writer.writeheader()

        # Write the new row
        writer.writerow({"position_name": position_name, "company": company})

    logger.info(f"Added URL {position_name} at {company} to blacklist")
    return {"message": f"{position_name} at {company} added to blacklist"}
    

@app.post('/submit-data')
def append_datapoint(data: dict):
    logger.info("Append datapoint endpoint called")
    datapoint_dict = generate_dataset_input(data['url'])
    datapoint_dict['rating'] = int(data['rating'])
    req_quals = extract_skills(datapoint_dict['posting_text'])
    datapoint_dict['req_quals'] = req_quals
    update_embeddings(req_quals)
    logger.info("successfully scraped and generated dataset input")

    # Create a new DataFrame with the new row
    new_row_df = pd.DataFrame([datapoint_dict])

    dataset_df = pd.read_csv('app/user_data/dataset/myDataset.csv')
    
    df = pd.concat([dataset_df, new_row_df], ignore_index=True)

    df.to_csv('app/user_data/dataset/myDataset.csv', index=False)

    updated_df = pd.read_csv('app/user_data/dataset/myDataset.csv')
    last_row_dict = updated_df.iloc[-1].to_dict()

    # Assert that the last row of the DataFrame has all the same data as the dictionary object
    assert datapoint_dict == last_row_dict

    logger.info("successfully updated dataset with new input")
  
    #print(updated_df.tail())

    return {'message': "success"}


@app.post('/retrain-model')
def retrain_model():
    logger.info("Retrain model endpoint called")
    #results = train_linreg('app/user_data/dataset/myDataset.csv')
    train_rfc('app/user_data/dataset/myDataset.csv', evaluate=False)
    logger.info("Done with model training")
    return {'message': "success"}

@app.post('/retrain-model-evaluate')
def retrain_model_evaluate():
    logger.info("Retrain model evaluate endpoint called")
    #results = train_linreg('app/user_data/dataset/myDataset.csv')
    results = train_rfc('app/user_data/dataset/myDataset.csv')
    logger.info("Done with model training and evaluation")
    return results

'''this function was used for evaluating accuracy of predictions from linear regression model, but we
later on switched to a Random Forest classifier model which warrants normal scoring for accuracy, so
this function is currently not used in the application'''
def custom_scoring_function(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, 3)
    y_pred = [round(pred) for pred in y_pred]
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def display_scores(scores, metric='rounded_accuracy'):
    if metric == 'rounded_accuracy':
        logger.info('Accuracy of rounded and clipped predictions: results of 5-fold cross val')
    if metric == 'neg_mse':
        logger.info('Neg mean squared error scores from 5-fold cross val')
    if metric == 'neg_mae':
        logger.info('Neg mean absolute error scores from 5-fold cross val')
    logger.info(f"Scores: {scores}")
    logger.info(f"Mean: {scores.mean()}")
    logger.info(f"Standard Deviation: {scores.std()}")

def train_rfc(dataset_path, evaluate=True):
    dataset = pd.read_csv(dataset_path)
    logger.info("read dataset into pandas dataframe")

    X = dataset[['posting_text', 'min_salary', 'max_salary']]
    y = dataset['rating']
    le = LabelEncoder()
    y_train = le.fit_transform(y)
    logger.info("label encoded the rating (target) column")

    tfidf_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X['posting_text'])
    logger.info("Fit new TFIDF vectorizer to entire dataset")

    X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=X.index)
    logger.info("Converted new TFIDF matrix to dataframe")

    new_df = X.drop(columns='posting_text')
    X_train = pd.concat([new_df, X_train_tfidf_df], axis=1)
    logger.info("generated new training dataframe after concatenating all feature columns except 'posting_text' with the new TFIDF dataframe")

    model = RandomForestClassifier(max_depth=None,  min_samples_leaf=1, min_samples_split=2, n_estimators=300, bootstrap=False, criterion='entropy')#, class_weight='balanced')
    model.fit(X_train, y_train)
    logger.info("fit RFC model to data")
    
    joblib.dump(tfidf_vectorizer, 'app/user_data/models/tfidf_vectorizer.joblib')
    logger.info("saved new TFIDF vectorizer model parameters to 'app/user_data/models/tfidf_vectorizer.joblib'")

    joblib.dump(model, 'app/user_data/models/rf_clf.joblib')
    logger.info("saved new Random Forest model to 'app/user_data/models/rf_clf.joblib'")

    # Save column names to a CSV file
    pd.DataFrame(X_train.columns).to_csv('app/user_data/models/column_names.csv', index=False, header=False)

    if evaluate:
      results = evaluate_rfc_performance(model, X_train, y_train)
      return results


def evaluate_rfc_performance(model, X_train, y_train):    
    with parallel_backend('loky', n_jobs=-1):
        accuracy_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
        mse_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        mae_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=5)

    display_scores(accuracy_scores, 'rounded_accuracy')
    accuracy_avg = round(accuracy_scores.mean() * 100, 1)
    accuracy_std = round(accuracy_scores.std() * 100, 1)

    display_scores(mse_scores, 'neg_mse')
    mse_avg = round(-1 * mse_scores.mean(), 3)
    mse_std = round(mse_scores.std(), 3)

    display_scores(mae_scores, 'neg_mae')
    mae_avg = round(-1 * mae_scores.mean(), 3)
    mae_std = round(mae_scores.std(), 3)

    return {
        'accuracy_avg': accuracy_avg,
        'accuracy_std': accuracy_std,
        'mae_avg': mae_avg,
        'mae_std': mae_std,
        'mse_avg': mse_avg,
        'mse_std': mse_std
    }

def train_linreg(dataset_path):
    custom_scorer = make_scorer(custom_scoring_function, greater_is_better=True)

    dataset = pd.read_csv(dataset_path)
    logger.info("read dataset into pandas dataframe")

    X = dataset[['posting_text', 'min_salary', 'max_salary']]
    y = dataset['rating']
    le = LabelEncoder()
    y_train = le.fit_transform(y)
    logger.info("label encoded the rating (target) column")

    tfidf_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X['posting_text'])
    logger.info("Fit new TFIDF vectorizer to entire dataset")

    X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=X.index)
    logger.info("Converted new TFIDF matrix to dataframe")

    new_df = X.drop(columns='posting_text')
    X_train = pd.concat([new_df, X_train_tfidf_df], axis=1)
    logger.info("generated new training dataframe after concatenating all feature columns except 'posting_text' with the new TFIDF dataframe")

    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("fit LinearRegression model to data")

    with parallel_backend('loky', n_jobs=-1):
        accuracy_scores = cross_val_score(model, X_train, y_train, scoring=custom_scorer, cv=5)
        mse_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        mae_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=5)

    display_scores(accuracy_scores, 'rounded_accuracy')
    accuracy_avg = round(accuracy_scores.mean() * 100, 1)
    accuracy_std = round(accuracy_scores.std() * 100, 1)

    display_scores(mse_scores, 'neg_mse')
    mse_avg = round(-1 * mse_scores.mean(), 3)
    mse_std = round(mse_scores.std(), 3)

    display_scores(mae_scores, 'neg_mae')
    mae_avg = round(-1 * mae_scores.mean(), 3)
    mae_std = round(mae_scores.std(), 3)

    joblib.dump(tfidf_vectorizer, 'app/user_data/models/tfidf_vectorizer.joblib')
    logger.info("saved new TFIDF vectorizer model to parameters to 'app/user_data/models/tfidf_vectorizer.joblib'")

    joblib.dump(model, 'app/user_data/models/linreg_clf.joblib')
    logger.info("saved new linear regression model 'app/user_data/models/linreg_clf.joblib'")

    return {
        'accuracy_avg': accuracy_avg,
        'accuracy_std': accuracy_std,
        'mae_avg': mae_avg,
        'mae_std': mae_std,
        'mse_avg': mse_avg,
        'mse_std': mse_std
    }

@app.get("/dataset")
async def get_dataset():
    logger.info("get dataset endpoint called")
    df = pd.read_csv("app/user_data/dataset/myDataset.csv")
    data = df.to_dict(orient="records")
    logger.info("converted dataset into dictionary object before sending to frontend")
    return JSONResponse(content=data)

@app.get("/download-dataset")
def download_dataset():
    logger.info("download dataset endpoint called")
    file_path = 'app/user_data/dataset/myDataset.csv'
    return FileResponse(file_path, media_type='text/csv', filename='myDataset.csv')

@app.get("/download-tfidf-model")
def download_tfidf_model():
    logger.info("download TFIDF model endpoint called")
    file_path = 'app/user_data/models/tfidf_vectorizer.joblib'
    return FileResponse(file_path, media_type='application/octet-stream', filename='tfidf_vectorizer.joblib')

@app.get("/download-rf-model")
def download_rf_model():
    logger.info("download random forest model endpoint called")
    file_path = 'app/user_data/models/rf_clf.joblib'
    return FileResponse(file_path, media_type='application/octet-stream', filename='rf_clf.joblib')

@app.get("/download-all")
async def download_all():
    logger.info("download all endpoint called")
    # Create an in-memory byte stream to hold the zip file
    zip_data = io.BytesIO()

    # Create a ZipFile object with the in-memory byte stream
    with zipfile.ZipFile(zip_data, mode="w") as zip_file:
        # Add the dataset file to the 'dataset' folder in the zip file
        dataset_file_path = 'app/user_data/dataset/myDataset.csv'
        with open(dataset_file_path, "r", newline="") as dataset_file:
            dataset_content = dataset_file.read()
            zip_file.writestr("dataset/myDataset.csv", dataset_content)

        # Add the tfidf model file to the 'models' folder in the zip file
        zip_file.write("app/user_data/models/tfidf_vectorizer.joblib", arcname="models/tfidf_vectorizer.joblib")

        # Add the linear regression model file to the 'models' folder in the zip file
        zip_file.write("app/user_data/models/rf_clf.joblib", arcname="models/rf_clf.joblib")

    logger.info("wrote all data to zip file")
    # Seek to the beginning of the in-memory byte stream
    zip_data.seek(0)

    # Set the Content-Disposition header to force download
    content_disposition = "attachment; filename=user_data.zip"

    # Return the in-memory byte stream as a StreamingResponse with the appropriate media type
    return StreamingResponse(io.BytesIO(zip_data.read()), media_type="application/zip", headers={"Content-Disposition": content_disposition})


# Function to show feature importances
def plot_feature_importances():
    # Load the model
    model = joblib.load('app/user_data/models/rf_clf.joblib')

    # Load feature names
    column_names = pd.read_csv('app/user_data/models/column_names.csv', header=None)[0].tolist()

    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 30  # Select top 10 features

    # Plotting
    plt.figure(figsize=(14, 3))
    plt.title("Feature Importances")
    plt.bar(range(top_n), importances[indices][:top_n], align="center")
    plt.xticks(range(top_n), [column_names[i] for i in indices[:top_n]], rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot to a static directory
    plot_file = 'static/feature_importances.png'  # Adjust path as needed
    plt.savefig(plot_file)
    plt.close()

    return plot_file

@app.get("/feature-importances")
async def get_feature_importances():
    print("get feature importances endpoint triggered")
    plt_path = plot_feature_importances()
    return {'plt_path': plt_path}


os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] = 'llama3-70b-8192'
os.environ["OPENAI_API_KEY"] = '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'

extractor = Agent(
    role = "qualifications extractor",
    goal = "extract required qualifications from the job posting and return as a list with a predictable format",
    backstory = "You are an AI assistant whose job is to extract all required qualifications from job postings and return them as a list, with each item separated by a new line character. Each item should be a phrase of no more than 5 words. Please summarize the required qualifications as fully as you can with minimal redundancy.",
    verbose = True,
    allow_delegation = False
)

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from statistics import mode

if "gte_model" or "gte_tokenizer" not in os.listdir("app/user_data/models/"):
    gte_tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
    gte_model = AutoModel.from_pretrained("thenlper/gte-large")
    # Save the tokenizer and model using save_pretrained
    gte_tokenizer.save_pretrained('app/user_data/models/gte_tokenizer')
    gte_model.save_pretrained('app/user_data/models/gte_model')

else:
    gte_tokenizer = AutoTokenizer.from_pretrained('app/user_data/models/gte_tokenizer')
    gte_model = AutoModel.from_pretrained('app/user_data/models/gte_model')

#gte_model = joblib.load('app/user_data/models/gte_model.joblib')
#gte_tokenizer = joblib.load('app/user_data/models/gte_tokenizer.joblib')

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

""" def extract_quals_worker(df_shared_mem_name, shape, dtype, start_idx, end_idx, process_id):
    # Access the shared memory
    existing_shm = shared_memory.SharedMemory(name=df_shared_mem_name)
    np_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    # Modify the DataFrame slice
    df_cols = ['posting_text', 'rating', 'min_salary', 'max_salary', 'req_quals']
    df_slice = pd.DataFrame(np_array[start_idx:end_idx], columns=df_cols)

    for row in df_slice.itertuples(index=True, name='Pandas'):
        posting_text = row.posting_text
        
        extract_quals = Task(
            description = f"Extract required skills from the following job posting:\n\n'{posting_text}'",
            agent = extractor,
            expected_output = "Here is an example output: '3+ years experience embedded C\nExperience writing unit tests\n1+ year experience technical lead'",
        )

        crew = Crew(
            agents = [extractor],
            tasks = [extract_quals],
            verbose = 2
        )

        while True:
            try:
                result = crew.kickoff()
                break
            except Exception as e:
                print(e)
                print(e.args)
        
        req_quals = result.split("\n")
        print(f"Debug: req_quals for row {row.Index}: {req_quals}")
        
        df_slice.at[row.Index, 'req_quals'] = req_quals

    # Copy back the changes to the shared array
    np_array[start_idx:end_idx] = df_slice.to_numpy()
    
    # Close the shared memory
    existing_shm.close() """

def extract_skills(posting_text):
    extract_quals = Task(
        description = f"Extract required skills from the following job posting:\n\n'{posting_text}'",
        agent = extractor,
        expected_output = "Here is an example output: '3+ years experience embedded C\nExperience writing unit tests\n1+ year experience technical lead'",
    )

    crew = Crew(
        agents = [extractor],
        tasks = [extract_quals],
        verbose = 2
    )
    
    while True:
        try:
            result = crew.kickoff()
            break
        except Exception as e:
            print(e)
            print(e.args)

    req_quals = result.split("\n")
    print(f"Debug: req_quals for row {row.Index}: {req_quals}")

    return req_quals

def update_embeddings(req_quals):
    # Load the embeddings (from PyTorch tensor)
    embeddings = torch.load('app/user_data/models/embeddings.pt')
    embeddings_list = []
    batch_size = 32 # Adjust this to a value that works within your memory constraints
    len_all = len(req_quals)
    print(f"req_quals size is {len_all}")
    for i in range(0, len_all, batch_size):
        print(f"iter is {i}")
        batch_phrases = req_quals[i:i + batch_size]
        print(batch_phrases)
        batch_dict = gte_tokenizer(batch_phrases, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():  # Disable gradient tracking
            outputs = gte_model(**batch_dict)
        embeddings = torch.cat((embeddings, average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])), dim=0)

    print("Saving embeddings tensor")
    torch.save(embeddings, "app/user_data/models/embeddings.pt")

def top_quals_analysis():
    df = pd.read_csv("app/user_data/dataset/myDataset.csv")
    import ast
    df['req_quals'] = df['req_quals'].apply(ast.literal_eval)
    row_labels = []
    for row in df.itertuples(index=True, name='Pandas'):
        req_quals = row.req_quals
        for phrase in req_quals:
            row_labels.append(row.Index)
    
    print("generated row labels")

    # Flatten the lists of phrases into a single list
    all_phrases = [phrase for sublist in df['req_quals'] for phrase in sublist]
    '''
    print("about to go through gte tokenizer")
    batch_dict = gte_tokenizer(all_phrases, max_length=512, padding=True, truncation=True, return_tensors='pt')
    print("done with gte tokenizer")
    outputs = gte_model(**batch_dict)
    print("generated outputs")
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    print("got through average pool")'''

    embeddings = torch.load('app/user_data/models/embeddings.pt')
    
    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    print("got through normalization")

    # we need to think about how many clusters we want in our phrase embedding space
    # 1 < N < 1000
    # N=500?
    # we can always change N

    kmeans = KMeans(n_clusters=500, random_state=57)
    print("fitting embeddings to K Means")
    kmeans.fit(embeddings)
    print("done fitting embeddings to K Means")
    # 1. Save the KMeans model with joblib
    print("Saving KMeans model")
    joblib.dump(kmeans, 'app/user_data/models/kmeans_model.joblib')
    #centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    # Step 1: Group embeddings by their cluster labels
    cluster_members_dict = defaultdict(list)
    embeddings_list = embeddings.tolist()
    for label, embedding in zip(cluster_labels, embeddings_list):
        cluster_members_dict[label].append(embedding)  # Keep embeddings as lists or arrays
        
    data = {'phrase': all_phrases, 'row_label': row_labels, 'cluster_label': cluster_labels}
    #pdb.set_trace()
    clusters_df = pd.DataFrame(data)

    # Create the qual_clusters column if it doesn't exist
    if 'qual_clusters' not in df.columns:
        df['qual_clusters'] = [[] for _ in range(len(df))]
    
    print("created clusters dataframe")
    cluster_score_dict = {}
    for row in df.itertuples(index=True, name='Pandas'):
        clusters_in_row = clusters_df.loc[clusters_df['row_label'] == row.Index, 'cluster_label'].tolist()
        # Generate value counts
        df.at[row.Index, 'qual_clusters'] = clusters_in_row
        for cluster in set(clusters_in_row): # we just want to look at unique cluster numbers per row
            if cluster not in cluster_score_dict.keys():
                cluster_score_dict[cluster] = row.rating 
            else:
                cluster_score_dict[cluster] += row.rating
    clusters_ranked = sorted(cluster_score_dict.items(), key=lambda item: item[1], reverse=True)
    top_10_cluster_scores = clusters_ranked[:10]
    # Extract the keys from the top n items
    top_10_clusters = [item[0] for item in top_10_cluster_scores]

    # Step 2: Calculate the mode embedding for each cluster
    top_10_cluster_modes = {}
    top_10_cluster_phrases = {}
    # Subset the cluster_member_dict
    top_10_dict = {key: cluster_members_dict[key] for key in top_10_clusters if key in cluster_members_dict}
    for label, embeddings in top_10_dict.items():
        try:
            # Use scipy.stats.mode to find the mode along axis 0
            mode_embedding = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=embeddings)
            top_10_cluster_modes[label] = mode_embedding
            # Find indices of embeddings equal to embedding_to_find
            indices = [index for index, emb in enumerate(embeddings_list) if emb == mode_embedding]
            top_10_cluster_phrases[label] = all_phrases[indices[0]]
        except Exception as e:
            print(e)
            # Handle case where there's no unique mode (e.g., all unique embeddings)
            cluster_emb_rep = embeddings[0] #representative embedding
            top_10_cluster_modes[label] = cluster_emb_rep
            indices = [index for index, emb in enumerate(embeddings_list) if emb == cluster_emb_rep]
            top_10_cluster_phrases[label] = all_phrases[indices[0]]

    top_10_phrase_scores = {}

    for item in top_10_cluster_scores:
        # Get the corresponding phrase from top_10_cluster_phrases using the cluster_label
        phrase = top_10_cluster_phrases[item[0]]
    
        # Get the score corresponding to the cluster_label from top_10_cluster_scores
        score = item[1]
    
        # Map the phrase to the score in the phrase_to_score dictionary
        top_10_phrase_scores[phrase] = score

    print("done getting top 10 phrase scores")  

    return top_10_phrase_scores

""" def top_quals_analysis_multiproc():
    df = pd.read_csv("app/user_data/dataset/myDataset.csv")
    row_labels = []

    # Ensure the new column 'req_quals' exists with default empty lists
    df['req_quals'] = [[] for _ in range(len(df))]
    # Convert DataFrame to a NumPy array
    np_array = df.to_numpy()

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
    shm_np_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
    np.copyto(shm_np_array, np_array)

    # Define process ranges
    num_processes = multiprocessing.cpu_count()
    rows_per_process = len(df) // num_processes
    processes = []

    for i in range(num_processes):
        start_idx = i * rows_per_process
        end_idx = (i + 1) * rows_per_process if i != num_processes - 1 else len(df)
        p = Process(target=extract_quals_worker, args=(shm.name, np_array.shape, np_array.dtype, start_idx, end_idx, i+1))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Read the modified shared memory into a DataFrame
    new_df = pd.DataFrame(shm_np_array, columns=df.columns)

    new_df.to_csv('app/user_data/dataset/skillsExtractedDataset.csv', index=False)
    
    # Clean up shared memory
    shm.close()
    shm.unlink()
    
    logger.info("Qualifications Extraction Complete")

    for row in new_df.itertuples(index=True, name='Pandas'):
        req_quals = row.req_quals
        for phrase in req_quals:
            row_labels.append(row.Index)

    # Flatten the lists of phrases into a single list
    all_phrases = [phrase for sublist in new_df['req_quals'] for phrase in sublist]
    
    batch_dict = tokenizer(all_phrases, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # we need to think about how many clusters we want in our phrase embedding space
    # 1 < N < 1000
    # N=500?
    # we can always change N

    kmeans = KMeans(n_clusters=500, random_state=57)
    kmeans.fit(embeddings)
    #centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    # Step 1: Group embeddings by their cluster labels
    cluster_members_dict = defaultdict(list)
    embeddings_list = embeddings.tolist()
    for label, embedding in zip(cluster_labels, embeddings_list):
        cluster_members_dict[label].append(embedding)  # Keep embeddings as lists or arrays
        
    data = {'phrase': all_phrases, 'row_label':` row_labels, 'cluster_label': cluster_labels}
    clusters_df = pd.DataFrame(data)

    cluster_score_dict = {}
    for row in new_df.itertuples(index=True, name='Pandas'):
        clusters_in_row = clusters_df.loc[clusters_df['row_label'] == row.Index, 'cluster_label'].tolist()
        # Generate value counts
        new_df.loc[row.Index, 'qual_clusters'] = clusters_in_row
        for cluster in set(clusters_in_row): # we just want to look at unique cluster numbers per row
            if cluster not in cluster_score_dict.keys():
                cluster_score_dict[cluster] = row.rating 
            else:
                cluster_score_dict[cluster] += row.rating
    clusters_ranked = sorted(dictionary.items(), key=lambda item: item[1], reverse=True)
    top_10_cluster_scores = clusters_ranked[:10]
    # Extract the keys from the top n items
    top_10_clusters = [item[0] for item in top_10_cluster_scores]

    # Step 2: Calculate the mode embedding for each cluster
    top_10_cluster_modes = {}
    top_10_cluster_phrases = {}
    # Subset the cluster_member_dict
    top_10_dict = {key: cluster_members_dict[key] for key in top_10_clusters if key in cluster_members_dict}
    for label, embeddings in top_10_dict.items():
        try:
            # Use scipy.stats.mode to find the mode along axis 0
            mode_embedding = mode(embeddings, axis=0).mode[0]
            top_10_cluster_modes[label] = mode_embedding
            # Find indices of embeddings equal to embedding_to_find
            indices = [index for index, emb in enumerate(embeddings_list) if emb == mode_embedding]
            top_10_cluster_phrases[label] = all_phrases[indices[0]]
        except StatisticsError:
            # Handle case where there's no unique mode (e.g., all unique embeddings)
            cluster_emb_rep = embeddings[0] #representative embedding
            top_10_cluster_modes[label] = cluster_emb_rep
            indices = [index for index, emb in enumerate(embeddings_list) if emb == cluster_emb_rep]
            top_10_cluster_phrases[label] = all_phrases[indices[0]]

    top_10_phrase_scores = {}

    for cluster_label in top_10_cluster_scores.keys():
        # Get the corresponding phrase from top_10_cluster_phrases using the cluster_label
        phrase = top_10_cluster_phrases[cluster_label]
    
        # Get the score corresponding to the cluster_label from top_10_cluster_scores
        score = top_10_cluster_scores[cluster_label]
    
        # Map the phrase to the score in the phrase_to_score dictionary
        top_10_phrase_scores[phrase] = score

    return top_10_phrase_scores """

@app.post("/insights")
async def get_insights():
    print("get insights endpoint triggered")
    top_10_phrase_scores = top_quals_analysis()
    print(top_10_phrase_scores)
    #return {"message": "success"}
    return top_10_phrase_scores






    





    


    


    