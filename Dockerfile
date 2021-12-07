Dockerfile
# Here is the build image
FROM python:3.8-alpine

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#gcc compliler required for pystan/prophet
RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get install g++ -y \
    && apt-get install build-essential -y \
    && apt-get clean

RUN pip install --upgrade pip
COPY requirements.txt . 
COPY requirements-ts.txt .
COPY requirements-test.txt .
COPY requirements-optional.txt .

#install requirements and files
RUN pip install -r requirements.txt
RUN pip install -r requirements-ts.txt
RUN pip install -r requirements-test.txt
RUN pip install -r requirements-optional.txt


#then copy the source/repo
COPY . .
