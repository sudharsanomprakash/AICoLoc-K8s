FROM python:3.9
RUN pip install flask prometheus-api-client
COPY . /app
WORKDIR /app
CMD ["python", "collector.py"]
