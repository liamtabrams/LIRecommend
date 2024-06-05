# Note: Important Project Status Update #

## Please Read Before Proceeding ##

### Attention Visitors and Potential Contributors ###

This repository is currently in a transitional phase, and there are critical updates underway regarding the security and stability of the project. As of now, there is a pending TODO which is to implement robust measures to safeguard sensitive information, particularly the OpenAI API key utilized within this project.

### Request for Collaboration Hold: ###

**I kindly request that you refrain from cloning or pushing to this repository until further notice, but it will remain public for educational and promotional purposes. This precautionary measure is essential to make sure the app can continue to work for everyone while the repo is in the state it's in currently.**

### Only 1 Option Right Now to Play With App: ###

In the meantime, if you wish to explore or utilize the lirecommend application, please reach out to me directly at **liamtabrams@gmail.com** to request access to the necessary application files, including the Docker build .tar file and the user_data folder. I would be more than happy to assist you and provide the required resources for your experimentation or usage of this tool.

## Stay Updated!!! ##

I appreciate your patience and understanding as I try to balance and weigh the following priorities: submitting this as my capstone project in order to complete the UCSD ML Engineering Bootcamp, seeking employment, and further developing and improving on this project. Please stay tuned for updates regarding the availability of this repository for collaborative contributions.

Thank you for your cooperation and continued interest in my project.

# LIRecommend

This README provides instructions on how to run LIRecommend using Docker.

## Prerequisites

- **Docker**: Ensure Docker is installed on your machine. If not, follow the instructions below to install it.

## Installing Docker

### Windows and macOS

1. Download Docker Desktop from the [Docker website](https://www.docker.com/products/docker-desktop).
2. Run the installer and follow the on-screen instructions.
3. After installation, start Docker Desktop.

### Linux

Follow the official Docker installation guide for your specific Linux distribution: [Docker Installation Guide](https://docs.docker.com/engine/install/).

## Running the Application

1. **Download the Docker Image File (or build)**

   Ensure you have `lirecommend.tar` file available on your machine, and if you would like to start off with using the pregenerated models and dataset, make sure to save the user_data folder provided in this repo to a desired location on your local machine, and copy the path to where you chose to save it. 'lirecommend.tar' will be distributed upon request since the file is too large to be stored in this repo. When this repo is made public, users should be able to clone the repo and build the LIRecommend Docker image themselves by running
   ```bash
   docker build -t lirecommend .
   ```
in the root directory of the cloned repository.

2. **Ensure user_data is prepared properly**
   You need to have a folder on your computer called 'user_data' that contains the following subfolders: 'blacklist', 'dataset', and 'models'. And those folders need to contain specific files. Thus, copy the user_data folder provided in this repo to a desired location on your local machine and note the absolute path. If you do not want to customize the dataset prior to launching the app for the first time, user_data should be ready for you to volume mount to the container you will run the app in (the volume mounting will be fully explained in Step 4). If you do want to do something like clear the dataset or adjust some of the labels for some of its existing datapoints (which you are allowed to do at any time), you can do that by directly modifying the dataset.csv file that is inside the dataset folder in 'user_data'. Just make sure not to change the structure of the CSV file or types and ranges of data being used in each column of the csv file, as well as to not corrupt the information that currently exists in the feature columns (non-rating columns) for the kept datapoints.   

3. **Load the Docker Image (if not building from sandbox)**

   Open a terminal or command prompt and navigate to the directory containing `lirecommend.tar`. Run the following command to load the Docker image:

   ```bash
   docker load -i lirecommend.tar
   ```

   This command loads the Docker image from the lirecommend.tar file into your local Docker repository.

4. **Run the Docker Container**

   After loading or building the image, start the Docker container using the following command.

   ```bash
   docker run -v /absolute/path/to/user_data:/code/app/user_data --name lirecommend-app -p 8000:8000 lirecommend
   ```
   -v will mount your locally stored user_data directory (the path of which precedes ':') to the directory in the Docker container with the path that follows ':', and this volume will remain mounted until explicitly removed (which is not necessary) or overwritten the next time docker -v is run with the paths. The point being is that you won't have to worry about keeping your local user_data up-to-date while using the app, as this makes the files you mounted from your machine to the Docker service (i.e. user_data) stay in sync with any updates or modifications that get saved in the container mountpoint. This allows you to NOT have to go through the following steps to save data as you are constructing your dataset and training models in: manually downloading (either through the web app by clicking 'Download All' or Docker Desktop) user_data from the container, and copying that folder back to the same place each time. If a user's container is closed or deleted, or a user restarts their computer, they won't lose the data they generated from using the app.   
   --name: The desired name for the Docker container; 'lirecommend-app' provided as example name. 
   -p 8000:8000: Map port 8000 on your host to port 8000 in the container. Adjust the ports if necessary.

5. **Access the Application**

    Open a Chrome browser (another browser MIGHT work but is not supported by LIRecommend currently) and go to http://localhost:8000/static/landing_page.html to access the application.



   
