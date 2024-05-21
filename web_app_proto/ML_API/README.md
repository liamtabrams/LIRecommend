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

   Ensure you have `lirecommend.tar` file available on your machine. This file should be included in the repository or provided separately.

2. **Load the Docker Image**

   Open a terminal or command prompt and navigate to the directory containing `lirecommend.tar`. Run the following command to load the Docker image:

   ```bash
   docker load -i lirecommend.tar
   ```

   This command loads the Docker image from the lirecommend.tar file into your local Docker repository.

3. **Run the Docker Container**

   After loading the image, start the Docker container using the following command:

   ```bash
   docker run -d -p 8080:80 lirecommend:latest
   ```

   -d: Run the container in detached mode (in the background).
   -p 8080:80: Map port 8080 on your host to port 80 in the container. Adjust the ports if necessary.

3. **Access the Application**

    Open a Chrome browser (another browser MIGHT work but is not supported by LIRecommend currently) and go to http://localhost:8080 to access the application.



   