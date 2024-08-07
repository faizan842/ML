# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (including gcc for compiling some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and install packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Define environment variables
ENV S3_BUCKET=your-s3-bucket-name
ENV S3_REGION=your-s3-region
ENV AWS_ACCESS_KEY_ID=your-aws-access-key-id
ENV AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key

# Command to run the app
CMD ["python", "app.py"]
