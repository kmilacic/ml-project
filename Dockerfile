FROM python:3.6.12

COPY . /app

WORKDIR /app

RUN apt-get update

RUN apt-get install -y libgl1-mesa-dev 'ffmpeg' 'libsm6' 'libxext6'

RUN pip install -r requirements.txt

CMD [ "flask", "run", "--host=0.0.0.0"]