# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install any needed packages specified in requirements.txt
# Assuming 'make requirements' installs dependencies from a requirements.txt or similar
RUN pip install --upgrade pip && \
    pip install setuptools wheel
RUN make requirements

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["bash"]
