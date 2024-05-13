FROM python:3.10

# Set up working directory
WORKDIR /code

# Copy requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the application code
COPY ./app /code/app

# Copy the static files
COPY ./static /code/static

# Expose port
EXPOSE 8000

# Define the command to run the application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]