# Use the official Python base image
FROM python:3.11-slim

# Set the working directory in the container
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install the libgomp1 package
RUN apt-get update && apt-get install -y libgomp1

# Copy the project files to the container
COPY . .

# Expose the port that the Dash app will run on (modify if needed)
ENV PORT 8080
EXPOSE $PORT

# Start the Dash app
CMD ["python", "app_final.py"]
