FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and source code
COPY models/ models/
COPY src/ src/
COPY config.yaml .

# Install additional monitoring dependencies
RUN pip install --no-cache-dir prometheus_client redis

# Expose ports for the API and monitoring
EXPOSE 5000 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models

# Create healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run both the API and monitoring server
CMD ["sh", "-c", "python -c 'from src.monitoring import ModelMonitor; monitor = ModelMonitor(); monitor.start_monitoring_server(8000)' & python src/serve.py"]