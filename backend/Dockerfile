FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads

# Create the app package directory and copy files
RUN mkdir -p app
COPY app/__init__.py app/
COPY app/config.py app/
COPY app/medical_analyzer.py app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODULE_NAME=main
ENV VARIABLE_NAME=app
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
