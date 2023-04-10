# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

#install stopwords and punkt
RUN pip install nltk
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Expose port 5000 for the Flask API
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]

