FROM python:3.7-slim

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install -y libgomp1

RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN pip install Flask

ENTRYPOINT ["python"]

CMD ["/app/main.py"]