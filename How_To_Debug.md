# Debugging Instructions for FastAPI Application in Docker

To debug the FastAPI application running in a Docker container, you can use the command line to view logs or access the log file directly using Docker Desktop. Follow the instructions below for both methods:

## Method 1: Using Command Line
   
   Use the following two commands to get the correct container_id and then view the logs of that container.

   ```bash
   docker ps -a
   docker logs <container_id>
   ```

## Method 2: Save and view log files on local machine
   
   - Open Docker Desktop on your local machine. 
   - Find the container running your FastAPI application. Containers are usually listed by their name or image. Click on the container to select it.
   - Click on the Files tab to access the container’s file system. Navigate to the code/app/logs directory to find the app.log file.
   - To simply view the log file within Docker Desktop, right-click on the app.log file and click 'Edit'. 
   - To view/save the log file on your local machine instead right-click on the app.log file and click 'Save' and choose the desired location on your machine where you'd like to save the log file. 