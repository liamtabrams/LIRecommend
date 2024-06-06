# LIRecommend

![Image 6-5-24 at 10 04 PM](https://github.com/liamtabrams/LIRecommend/assets/34357178/4267a5b6-a478-443d-b802-f0d7652b785d)

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

![Image 6-5-24 at 10 23 PM](https://github.com/liamtabrams/LIRecommend/assets/34357178/01f75293-7e37-41d2-ba1a-8e0c2d23492a)

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

![Image 6-5-24 at 10 29 PM](https://github.com/liamtabrams/LIRecommend/assets/34357178/86296cae-4c3f-442d-b643-62bbfcd1fe18)

![Image 6-5-24 at 10 26 PM](https://github.com/liamtabrams/LIRecommend/assets/34357178/654d36a2-77b4-49e6-af7a-bfe5eb592f2c)

![Image 6-5-24 at 10 24 PM](https://github.com/liamtabrams/LIRecommend/assets/34357178/58976bfb-19fd-4e0f-adff-0810a13a32cf)

### FOR MORE INFO ###

see https://github.com/liamtabrams/LIRecommend/blob/main/archive/README.md for an overview of the vision and goals behind this project as well as some of the implementation details, plus summaries of the most noteworthy files and directories in this repository. 
   
