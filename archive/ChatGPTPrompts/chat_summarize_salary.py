# getting started with OpenAI API!

import os
from openai import OpenAI
import json

def generate_salary_json_file(ind, location_val):

  prompt = "Take the following python data and infer the minimum and maximuim of the salary range, and fill out their values as floating point numbers with three decimal places in units of thousands in a JSON dictionary, with 'salary_min' and 'salary_max' being the keys. If the data given is ['$150,000.00/yr - $220,000.00/yr'] then you should return {'salary_min': 150, 'salary_max': 220}, but make sure to use double quotes to enclose the key names. If you infer that info is in dollars per hour, convert the numbers to annual salary in thousands so output is same regardless of given units. Note that $48/hr is equal to $100,000/yr. Put 'N/A' under a fields if the required information is not given. If only one number is given put it under 'salary_max'. Return only the JSON dictionary. I want you to do it, not to tell me how to code it. I want you to do it for: "

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
  json_files_dir = r"drive/MyDrive/jobSalaries/"
  print(chat_completion.choices[0].message.content)
  json_data = json.loads(chat_completion.choices[0].message.content.strip("`").strip('json').strip())
  json_file_path = json_files_dir + f'row{ind + 1}.json'
  # Write JSON data to the file
  with open(json_file_path, "w") as json_file:
    json.dump(json_data, json_file)

  return json_file_path

  for index, value in enumerate(df['salary/compensation_range']):
    generate_salary_json_file(index, value)