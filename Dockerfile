FROM python:3.9-slim

# Set the working directory
WORKDIR /app

RUN curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth ./models/

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code into the container
COPY . .

# Expose the Flask default port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]