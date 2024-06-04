# getting started with OpenAI API!

import os
from openai import OpenAI
import json

def generate_location_json_file(ind, location_val):

  prompt = "Take the following python list and create a JSON dictionary containing 'city', 'state', and 'country' as keys and fill out the values. If the list is ['Colorado Springs, CO'] you would return {'city': 'Colorado Springs', 'state': 'CO', 'country': 'USA'}. If the list is instead Return only the JSON dictionary, and make sure to use double quotes even though I used single in this prompt. I want you to do it, not to tell me how to code it. I want you to do it for: "

  prompt = prompt + "/n" + str(location_val)

  client = OpenAI(
    # This is the default and can be omitted
    api_key="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",
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
  json_files_dir = r"drive/MyDrive/jobLocations/"
  print(chat_completion.choices[0].message.content)
  json_data = json.loads(chat_completion.choices[0].message.content.strip("`").strip('json').strip())
  json_file_path = json_files_dir + f'row{ind + 1}.json'
  # Write JSON data to the file
  with open(json_file_path, "w") as json_file:
    json.dump(json_data, json_file)

  return json_file_path

for index, value in enumerate(df['location']):
  generate_location_json_file(index, value)