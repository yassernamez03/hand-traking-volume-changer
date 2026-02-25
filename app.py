import cv2
import mediapipe as mp
import time
from pycaw.pycaw import AudioUtilities
import math
import numpy as np
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


class handDetector():
    def __init__(self, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.results = None

    def findHands(self, img, Draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.landmarker.detect(mp_image)

        if self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                if Draw:
                    h, w, c = img.shape
                    # Draw landmarks
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    # Draw connections
                    for connection in HAND_CONNECTIONS:
                        start = hand_landmarks[connection.start]
                        end = hand_landmarks[connection.end]
                        sx, sy = int(start.x * w), int(start.y * h)
                        ex, ey = int(end.x * w), int(end.y * h)
                        cv2.line(img, (sx, sy), (ex, ey), (0, 255, 0), 2)
        return img

    def findPosition(self, img, draw=True):
        Plist = []
        if self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                for id, lm in enumerate(hand_landmarks):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    Plist.append([id, cx, cy])
        return Plist


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    devices = AudioUtilities.GetSpeakers()
    volume = devices.EndpointVolume
    volbar = 400
    volper = 0

    volMin, volMax = volume.GetVolumeRange()[:2]
    while True:
        success, img = cap.read()
        if not success:
            continue
        img = detector.findHands(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        Plist = detector.findPosition(img)

        if Plist != []:
            x1, y1 = Plist[4][1], Plist[4][2]
            x2, y2 = Plist[8][1], Plist[8][2]
            cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            length = math.hypot(x2 - x1, y2 - y1)
            print(length)

            vol = np.interp(length, [30, 350], [volMin, volMax])
            volbar = np.interp(length, [30, 350], [400, 150])
            volper = np.interp(length, [30, 350], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
            cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
