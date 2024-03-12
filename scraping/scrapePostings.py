import requests
from bs4 import BeautifulSoup
import csv
import time

def scrape_job_postings(csv_file):
    # Open the CSV file
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        #csv_writer = csv.writer(file)
        row_num = 1

        for row in reader:
          text_file_path = f"text_files2/row{row_num}.txt"
          with open(text_file_path, 'w') as file:
            job_link = row[0] # Assuming the job links are in the first column
            print(job_link)
            # Send a GET request to the job link
            # request web page
            resp = requests.get(fr"{job_link}")

            if resp.status_code == 200:
              # get the response text. in this case it is HTML
              html = resp.text
              # Parse the HTML content
              soup = BeautifulSoup(html, 'html.parser')

            else:
              time.sleep(1)
              # request again
              resp = requests.get(job_link)
              if resp.status_code == 200:
                # get the response text. in this case it is HTML
                html = resp.text
                # Parse the HTML content
                soup = BeautifulSoup(html, 'html.parser')

              else:
                print("Failed to retrieve job posting from", job_link)
                row_num += 1
                continue

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
          row_num += 1



# Example usage
#csv_file_path = "job_links.csv"
#scrape_job_postings(csv_file_path)