import json
import requests

data = {"features": ["https://www.linkedin.com/jobs/view/3917257078/?alternateChannel=search&refId=ZMANnVNE15NLEGtNemxpkw%3D%3D&trackingId=mkrEETO8St6Tm7udlXjJnQ%3D%3D"]}

url = 'http://0.0.0.0:8000/predict'

data = json.dumps(data)
response = requests.post(url, data)
print(response.json())