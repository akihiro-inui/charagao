FROM ubuntu:latest

RUN apt-get update
RUN apt-get install python3 python3-pip -y

# COPY stuff, directory path setting
ADD . /app
WORKDIR /app
ENV PYTHONPATH /app
ENV STATIC_URL src/static
ENV STATIC_PATH src/static

# Download libraries
RUN pip3 install -r src/requirements.txt

# Port forward
EXPOSE 5000

# Run web app
CMD python3 src/app.py
