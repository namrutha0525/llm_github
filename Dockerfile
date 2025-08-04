# Use a standard Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage caching
COPY requirements.txt .

# Install all the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Tell Hugging Face that the app will run on port 7860
EXPOSE 7860

# The command to start your Uvicorn server for final_app.py
# It must listen on host 0.0.0.0 and port 7860
CMD ["uvicorn", "final_app:app", "--host", "0.0.0.0", "--port", "7860"]
