FROM anibali/pytorch:1.13.0-cuda11.8
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /classification