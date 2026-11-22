FROM python:3.11-slim

# Avoid prompts from debian frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=5001
ENV QT_QPA_PLATFORM=offscreen
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# Install system dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python || true && \
    pip install --no-cache-dir opencv-python-headless

# Copy source code and initial directories
COPY . .

# Ensure upload and reports directories exist
RUN mkdir -p REPORTS/daily REPORTS/weekly REPORTS/monthly AI_MODELS

# Expose port (Railway sets $PORT dynamically)
EXPOSE 5001

# Run with gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5001} server:app --workers 1 --threads 4 --timeout 120"]
