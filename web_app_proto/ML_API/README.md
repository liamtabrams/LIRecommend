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

1. **Download the Docker Image File**

   Ensure you have `lirecommend.tar` file available on your machine, and if you would like to start off with using the pregenerated models and dataset, extract and copy the user_data zip file provided with lirecommend to an appropriate folder on your machine and copy the absolute path to that directory. These files should be included in the repository or provided separately.

2. **Load the Docker Image**

   Open a terminal or command prompt and navigate to the directory containing `lirecommend.tar`. Run the following command to load the Docker image:

   ```bash
   docker load -i lirecommend.tar
   ```

   This command loads the Docker image from the lirecommend.tar file into your local Docker repository.

3. **Run the Docker Container**

   After loading the image, start the Docker container using the following command.

   ```bash
   docker run -v /absolute/path/to/user_data:/code/app/user_data --name lirecommend-app -p 8000:8000 lirecommend
   ```
   -v will mount your locally stored user_data directory (the path of which precedes ':') to the directory in the Docker container with the path that follows ':', and this volume will remain mounted until explicitly removed (which is not necessary) or overwritten the next time docker -v is run with the paths. The point being is that you won't have to worry about keeping your local user_data up-to-date while using the app, as this makes the files you mounted from your machine to the Docker service (i.e. user_data) stay in sync with any updates or modifications that get saved in the container mountpoint. This allows you to NOT have to go through the following steps to save data as you are constructing your dataset and training models in: manually downloading (either through the web app by clicking 'Download All' or Docker Desktop) user_data from the container, and copying that folder back to the same place each time. If a user's container is closed or deleted, or a user restarts their computer, they won't lose the data they generated from using the app.   
   --name: The desired name for the Docker container; 'lirecommend-app' provided as example name. 
   -p 8000:8000: Map port 8000 on your host to port 8000 in the container. Adjust the ports if necessary.

3. **Access the Application**

    Open a Chrome browser (another browser MIGHT work but is not supported by LIRecommend currently) and go to http://localhost:8000/static/landing_page.html to access the application.



   