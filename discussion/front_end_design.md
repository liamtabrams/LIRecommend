For the capstone project, I want to essentially demo my own instance of a product/service which allows users to archive descriptions of job postings into bins based on ratings they give them and have them returned recommendations for unviewed job postings as well as other insights that can be obtained from the data using machine learning and statistical analysis. 

This may entail a user login page for authentication into the web service, so that a given user can load his/her unique profile and data associated it, which includes their job preference model parameters. 

We have 2 general options for deployment:
1. deploy as web app which will require using a server and database provided by a Cloud computing service.
2. deploy app using a Docker container

In the first scenario data is stored in a web server or database, while in the second scenario the data is stored locally on the user's computer. 

Once the user is authenticated/logged in (which will be different depending on whether the app is deployed within a Docker container or by a paid web-hosting service) the landing page should have a menu of the following options to click on: "Collect more data", "View rated job postings", "Insights", and "Recommend". If the user clicks on "Collect more data", then another page will open that has a text box next to "enter url" and a set of buttons ranging from 0 to 3 stars underneath the text box for the user to select from to rate the given job posting.  
