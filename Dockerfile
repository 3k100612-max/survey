# Use a lightweight Python image
FROM python:3.11-slim

# Install system dependencies for PostgreSQL (psycopg2)
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_PORT=8506
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 8507

# Start the application
CMD ["flask", "run"]
