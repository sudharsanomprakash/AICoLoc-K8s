# Use official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir torch flask prometheus-api-client

# Copy source code and model file
COPY model.py inferencer.py model.pth ./

# Expose Flask API port
EXPOSE 5000

# Run the AI inference service
CMD ["python", "inferencer.py"]
