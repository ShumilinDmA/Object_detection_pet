FROM python:3.8
MAINTAINER Dmitrii Shumilin <ShumilinDmAl@gmail.com>

RUN pip install --upgrade pip
WORKDIR /usr/scr/application

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8181

CMD ["python", "web_app.py"]