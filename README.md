# Hand Tracking Volume Changer

Control your system volume with hand gestures using your webcam. Pinch your thumb and index finger together to lower the volume, spread them apart to raise it.

Built with OpenCV, MediaPipe, and pycaw.

## Requirements

- Python 3.12+
- Webcam

## Setup

```bash
pip install -r requirements.txt
```

Download the MediaPipe hand landmarker model:

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Usage

```bash
python app.py
```

- Show your hand to the webcam
- Move your **thumb** and **index finger** closer or further apart to control volume
- A volume bar on screen shows the current level
- Press **q** to quit

## Docker

```bash
docker compose up --build
```

Requires webcam, display, and audio passthrough — see `docker-compose.yml` for details.
