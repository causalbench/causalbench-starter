FROM alpine:latest

WORKDIR /code

COPY requirements.txt .
COPY src/ .
