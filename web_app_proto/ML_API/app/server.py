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
import io
import zipfile
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates as templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import parallel_backend

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
  try:
    text_file_path, posting_ind = scrape_from_link(url)
  except Exception as e:
    logger.error(f"Encountered error {e} when trying to scrape from {url}")
    return
  
  with open(text_file_path, "r") as f:
    file_contents = f.read()
    f.seek(0)
    lines = f.readlines()

  datapoint_dict = {}
  salary_line = lines[3]
  salary_val = salary_line.strip("\n").strip("salary is ")
  logger.info(f"attempting to generate JSON salary file for {url} using ChatGPT API")
  try:
    json_file_path = generate_salary_json_file(posting_ind, salary_val)
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
    red = 255 - green
    
    # Return the color as a hex code
    return f'#{red:02x}{green:02x}{blue:02x}'

def generate_prediction(url):
    while True:
        try:
            datapoint = generate_dataset_input(url)
            logger.info("generated datapoint")
            model = joblib.load('app/user_data/models/linreg_clf.joblib')
            tfidf_vect = joblib.load('app/user_data/models/tfidf_vectorizer.joblib')
            logger.info("loaded pre-trained Linear Regression and TFIDF Vectorizer models")
            # Step 3: Transform the test data using the fitted vectorizer
            X_test_tfidf = tfidf_vect.transform([datapoint['posting_text']])
            logger.info("transformed datapoint's posting_text into TFIDF matrix")
            # Convert TF-IDF matrix to DataFrame
            X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vect.get_feature_names_out())
            logger.info("transformed the matrix into a dataframe")
            #Create DataFrame excluding 'posting_text' column
            new_df = pd.DataFrame({key: [value] for key, value in datapoint.items() if key != 'posting_text'})
            X_test = pd.concat([new_df, X_test_tfidf_df], axis=1)
            logger.info("concatenated original columns minus posting_text with TFIDF dataframe")
            prediction = model.predict(X_test)
            logger.info(f"generated prediction {prediction} with the following datapoint: {X_test}")
            color = get_rating_color(prediction)
            logger.info("generated prediction and color code")
            return prediction[0], color, datapoint
        except:
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
    return {'prediction': round(prediction, 3), 'color': color}

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
            job["rating"] = round(result[0], 2)
            position_line = result[2]["posting_text"].split('\n')[0]
            company_line = result[2]["posting_text"].split('\n')[1]
            position = position_line.strip("position is ")
            company = company_line.strip("company is ")
            job["position"] = position
            job["company"] = company
            job["color"] = result[1]
            recommendations.append(job)

    # Sort the job postings by rating (highest first)
    recommendations.sort(key=lambda x: x["rating"], reverse=True)

    # Return the top 10 job postings
    return JSONResponse(content={"recommendations": recommendations[:10]})
    

@app.post('/submit-data')
def append_datapoint(data: dict):
    logger.info("Append datapoint endpoint called")
    datapoint_dict = generate_dataset_input(data['url'])
    datapoint_dict['rating'] = int(data['rating'])
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


@app.post('/retrain-model')
def retrain_model():
    logger.info("Retrain model endpoint called")
    results = train_linreg('app/user_data/dataset/myDataset.csv')
    logger.info("Done with model evaluation")
    return results

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
    logger.info("saved new Linear Regression model 'app/user_data/models/linreg_clf.joblib'")

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
    file_path = 'app/user_data/dataset/myDataset.csv'
    return FileResponse(file_path, media_type='text/csv', filename='myDataset.csv')

@app.get("/download-tfidf-model")
def download_tfidf_model():
    file_path = 'app/user_data/models/tfidf_vectorizer.joblib'
    return FileResponse(file_path, media_type='application/octet-stream', filename='tfidf_vectorizer.joblib')

@app.get("/download-linreg-model")
def download_linreg_model():
    file_path = 'app/user_data/models/linreg_clf.joblib'
    return FileResponse(file_path, media_type='application/octet-stream', filename='linreg_clf.joblib')

@app.get("/download-all")
async def download_all():
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
        zip_file.write("app/user_data/models/linreg_clf.joblib", arcname="models/linreg_clf.joblib")

    # Seek to the beginning of the in-memory byte stream
    zip_data.seek(0)

    # Set the Content-Disposition header to force download
    content_disposition = "attachment; filename=user_data.zip"

    # Return the in-memory byte stream as a StreamingResponse with the appropriate media type
    return StreamingResponse(io.BytesIO(zip_data.read()), media_type="application/zip", headers={"Content-Disposition": content_disposition})