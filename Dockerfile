# Use an official Python runtime as a parent image
FROM python:3.10.6

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/
# COPY C:/Users/andkelly/Documents/Resumes/ /app/

# Install any needed packages specified in the requirements file
RUN pip install -r requirements.txt

# Copy your additional file into the container
COPY chat_over_docs.py /app/

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install vim -y

# Specify the command to run when the container starts
# CMD [ "python", "chat_over_docs.py" ]

