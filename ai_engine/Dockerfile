FROM python:3.10-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies including kubernetes
RUN pip install --no-cache-dir \
    torch==2.2.1 \
    flask==3.1.1 \
    prometheus-api-client \
    kubernetes \
    "numpy<2"

WORKDIR /app

COPY . .

EXPOSE 5000

CMD ["python", "inferencer.py"]

