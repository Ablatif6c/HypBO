# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY ./requirements/requirements.txt .
COPY ./requirements/requirements_dev.txt .
RUN pip install --upgrade pip
RUN python -m pip install -r requirements.txt
RUN python -m pip install -r requirements_dev.txt

WORKDIR /app
COPY . /app

# Entry point
CMD ["/bin/bash"]
