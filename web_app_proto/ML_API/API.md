# API Usage

To interact with LIRecommend using the APIs programmatically rather than the graphical web app, one can use the json and requests libraries in Python in order to get and post data from and to the API endpoints using JSON format. See the below example code for sending a job posting URL to the predict endpoint, assuming the application was launched using 'docker run --name "container-name" -p 8000:8000 lirecommend'. 

```python
import json
import requests

data = {"features": ["https://www.linkedin.com/jobs/view/3917257078/?alternateChannel=search&refId=ZMANnVNE15NLEGtNemxpkw%3D%3D&trackingId=mkrEETO8St6Tm7udlXjJnQ%3D%3D"]}

url = 'http://0.0.0.0:8000/predict'

data = json.dumps(data)
response = requests.post(url, data)
print(response.json()) #should print something like {'prediction': 1.2013245748404626, 'color': '#9e6138'}
```

This general method can be used to access the different API endpoints of the application. If we wanted to instead, say, submit a new datapoint and retrain the model programmatically, we could use the following code to do this 

```python
import json
import requests

data = {"url": r"https://www.linkedin.com/jobs/view/3925914578/?eBP=CwEAAAGPnVJq4Sql2Ejms8Dr3vRvAk4WQ4CPVxc6KhpcL4-xa-p4ZprNzPzCSUVO7TQFap_5q_B_Iq_-tyUpKBo19f5Htfrj7wBGwLGYjIeLMcTZOsYGPNksfQ2tsu7tllyHJXll_Al2SkKJZf4XgQs7lPTJ8hH_aBs_qZvvGdAujhoH5h7n2f_QdCxItR4nMFsCsaW57r9Jjjfx7GKPvf1Jsh2iUU6ebYIzaLespuKjk5HImQpAOUt5YSC29Fb20kwPWFEbNNVdNsAHc541e6cwmfPLeiNG8mC3BBFsOwMKp89EtsFMSpz6Z92tt36dAfKbDp-dfDvIB1basLcStm60zZozHg768CiEYgR8Du8fgX0EclH01kE68dWxT1swZgqSQ8kuvZ0-rUCLvr6ihepCr_ay&refId=hJLm0lmb2Bzthp5YAjBDzA%3D%3D&trackingId=F4J6Rg8PlEn%2FYbYFRIiomA%3D%3D&trk=flagship3_jobs_discovery_jymbii", "rating": 2}

url = 'http://0.0.0.0:8000/submit-data'

data = json.dumps(data)
response = requests.post(url, data)
print(response.json()) #should print None

url = 'http://0.0.0.0:8000/retrain-model'
data = {}
data = json.dumps(data)
response = requests.post(url, data)
print(response.json()) '''should print something like {'accuracy_avg': 59.324324324324316, 'accuracy_std': 11.96772051990481, 'mae_avg': 0.5564076573245048, 'mae_std': 0.12388109200691212, 'mse_avg': 0.627168855240376, 'mse_std': 0.48269694288801385}'''
```

See client.py. You can run that script on your local machine while the app is running to test that you have access to the API endpoints of the app. Just make sure you are using the correct base URL for the APIs. You will have to modify the URL's used in client.py which are all hard-coded if you chose something other than 8000:8000 for the -p option when you launched the app. 

