
##base image
FROM python:3.12-slim


# Set working directory
WORKDIR /app

# Copy files,model and preprocessor
COPY . /app

#Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# Expose port
EXPOSE 9696

# Run the app

ENTRYPOINT ["uvicorn", "main:app", "--host=0.0.0.0", "--port=9696"]

