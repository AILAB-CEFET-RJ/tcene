# Use official Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /tcene

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt


# Expose Streamlit default port
EXPOSE 8501

# Run your Streamlit app
CMD ["streamlit", "run", "app.py"]



# /mnt/c/Users/parai/Documents/Github - tcene/tcene$