# Use a lightweight Python image
FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Expose the port (matching the one used in ACI deployment)
EXPOSE 80

# Run the application (customize as needed)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
