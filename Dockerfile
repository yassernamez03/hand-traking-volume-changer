FROM python:3.12-slim

# System deps for OpenCV, mediapipe, display, and audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    alsa-utils \
    pulseaudio-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the hand landmarker model at build time so it's baked into the image
RUN curl -L -o hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

COPY app.py .

CMD ["python", "app.py"]
