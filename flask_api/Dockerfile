FROM python:3.6.9-stretch

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["uwsgi", "app.ini"]