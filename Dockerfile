# Use a lightweight official Python image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (CMake, gcc, g++, and common libraries for ML)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Expose the port (Ensure it matches ACI deployment)
EXPOSE 8080

# Run the application (customize as needed)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
