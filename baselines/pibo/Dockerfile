# Base image for Python environment
FROM continuumio/anaconda3:2021.05

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy project files to the container
COPY . /app

# # Install project dependencies
# RUN pip install . 

# During debugging, this entry point will be overridden
CMD ["python", "main.py"]
