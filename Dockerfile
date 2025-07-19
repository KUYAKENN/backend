# Use Python slim image - MediaPipe/TensorFlow deployment
FROM python:3.11-slim

# Install minimal system dependencies for MediaPipe and OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (no dlib!)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE $PORT

# Start command with proper environment variable handling
CMD ["sh", "-c", "gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload --log-level info"]
