# getting started with OpenAI API!
import os
from openai import OpenAI
import json

def generate_json_file(text_file_path):
  with open(text_file_path, 'r') as file:
    # Read the contents of the file into a string
    file_contents = file.read()

  prompt = "Take the following text from a job posting and extract all info related to the following fields which will end up being the keys in the JSON data you will return: emploment type, job function, description of product/service, industries, position name, broader role name, company, location, salary/compensation range, responsibilities, goals/objectives, name of department/team, required qualifications, preferred qualifications, benefits, work arrangement (hybrid, on-site, remote?), organizing this info into a table where the first column contains the fields and the second column contains the info relevant to these fields. Try to extract as much of the info as possible while being as succinct as possible. Remove redundancy wherever possible. If there is no info related to a particular field, put 'N/A'. Standardize entries in the info column by separating individual considerations with a hyphen. Do not exceed 5 words for any individual consideration in the info column. Use acronyms and abbreviations where necessary to do so. Return the table as a json data structure that is loadable in python, and make sure the keys are the fields listed earlier, not 'fields' and 'info'. Store the dictionary values as python lists. Return only the JSON and no title. Here is the text from the job posting: "

  prompt = prompt + "/n" + file_contents

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
  json_files_dir = r"drive/MyDrive/LI-Jobs-JSON/"
  json_data = json.loads(chat_completion.choices[0].message.content.strip("`").strip('json').strip())
  json_file_path = json_files_dir + text_file_path.strip(r"drive/MyDrive/text_files2/").strip('.txt') + '.json'
  # Write JSON data to the file
  with open(json_file_path, "w") as json_file:
    json.dump(json_data, json_file)

  return json_file_path
