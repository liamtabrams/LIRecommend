import json
import requests

# make a python dictionary
data = {"url": "https://www.linkedin.com/jobs/view/personal-assistant-at-insight-global-3937766346?position=2&pageNum=0&refId=sJXzlkIukBcy5NXp3x8bVg%3D%3D&trackingId=AtpylxKn25MVI210uE9ibw%3D%3D&trk=public_jobs_jserp-result_search-card"}

# this is the URL for the predict API
api_url = 'http://0.0.0.0:8000/predict'

# convert dictionary to correct JSON format
data = json.dumps(data)
# post the JSON data to the API endpoint
response = requests.post(api_url, data)
print(response.json()) #should print something like '''{'prediction': 0, 'color': '#c03131'}'''

data = {"url": r"https://www.linkedin.com/jobs/view/3925914578/?eBP=CwEAAAGPnVJq4Sql2Ejms8Dr3vRvAk4WQ4CPVxc6KhpcL4-xa-p4ZprNzPzCSUVO7TQFap_5q_B_Iq_-tyUpKBo19f5Htfrj7wBGwLGYjIeLMcTZOsYGPNksfQ2tsu7tllyHJXll_Al2SkKJZf4XgQs7lPTJ8hH_aBs_qZvvGdAujhoH5h7n2f_QdCxItR4nMFsCsaW57r9Jjjfx7GKPvf1Jsh2iUU6ebYIzaLespuKjk5HImQpAOUt5YSC29Fb20kwPWFEbNNVdNsAHc541e6cwmfPLeiNG8mC3BBFsOwMKp89EtsFMSpz6Z92tt36dAfKbDp-dfDvIB1basLcStm60zZozHg768CiEYgR8Du8fgX0EclH01kE68dWxT1swZgqSQ8kuvZ0-rUCLvr6ihepCr_ay&refId=hJLm0lmb2Bzthp5YAjBDzA%3D%3D&trackingId=F4J6Rg8PlEn%2FYbYFRIiomA%3D%3D&trk=flagship3_jobs_discovery_jymbii", "rating": 2}

url = 'http://0.0.0.0:8000/submit-data'

data = json.dumps(data)
response = requests.post(url, data)
print(response.json()) #should print {'message': "success"}

url = 'http://0.0.0.0:8000/retrain-model'
data = {}
data = json.dumps(data)
response = requests.post(url, data)
print(response.json()) #should print {'message': "success"}