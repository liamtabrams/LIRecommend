import json
import requests

data = {"features": ["https://www.linkedin.com/jobs/view/3917257078/?alternateChannel=search&refId=ZMANnVNE15NLEGtNemxpkw%3D%3D&trackingId=mkrEETO8St6Tm7udlXjJnQ%3D%3D"]}

url = 'http://0.0.0.0:8000/predict'

data = json.dumps(data)
response = requests.post(url, data)
print(response.json())

data = {"url": r"https://www.linkedin.com/jobs/view/3925914578/?eBP=CwEAAAGPnVJq4Sql2Ejms8Dr3vRvAk4WQ4CPVxc6KhpcL4-xa-p4ZprNzPzCSUVO7TQFap_5q_B_Iq_-tyUpKBo19f5Htfrj7wBGwLGYjIeLMcTZOsYGPNksfQ2tsu7tllyHJXll_Al2SkKJZf4XgQs7lPTJ8hH_aBs_qZvvGdAujhoH5h7n2f_QdCxItR4nMFsCsaW57r9Jjjfx7GKPvf1Jsh2iUU6ebYIzaLespuKjk5HImQpAOUt5YSC29Fb20kwPWFEbNNVdNsAHc541e6cwmfPLeiNG8mC3BBFsOwMKp89EtsFMSpz6Z92tt36dAfKbDp-dfDvIB1basLcStm60zZozHg768CiEYgR8Du8fgX0EclH01kE68dWxT1swZgqSQ8kuvZ0-rUCLvr6ihepCr_ay&refId=hJLm0lmb2Bzthp5YAjBDzA%3D%3D&trackingId=F4J6Rg8PlEn%2FYbYFRIiomA%3D%3D&trk=flagship3_jobs_discovery_jymbii", "rating": 2}

url = 'http://0.0.0.0:8000/submit-data'

data = json.dumps(data)
response = requests.post(url, data)
print(response.json())

url = 'http://0.0.0.0:8000/retrain-model'
data = {}
data = json.dumps(data)
response = requests.post(url, data)
print(response.json())