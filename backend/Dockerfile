FROM python:3.10-slim

WORKDIR /app

# (optional but good) install system libs opencv sometimes needs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy backend folder into container
COPY backend/ /app/backend/

WORKDIR /app/backend

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# Render provides PORT env var; default to 10000
CMD ["sh", "-c", "uvicorn app.api_server:app --host 0.0.0.0 --port ${PORT:-10000}"]
