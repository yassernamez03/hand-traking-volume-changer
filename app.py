import cv2
import mediapipe as mp
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import numpy as np



class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, Draw=True):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # process the frame
        #     print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if Draw:
                    # Draw dots and connect them
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    # def calDistance(self, img, draw=True):
    #     dis = 0
    #     cx1, cy1 = 0, 0
    #     cx2, cy2 = 0, 0
    #     if self.results.multi_hand_landmarks:
    #         for handLms in self.results.multi_hand_landmarks:
    #             for id, lm in enumerate(handLms.landmark):
    #                 h, w, c = img.shape
    #                 cx, cy = int(lm.x * w), int(lm.y * h)
    #                 if id == 4:
    #                     cx1, cy1 = cx, cy
    #                     cv2.circle(img, (cx1, cy1), 25, (255, 0, 255), cv2.FILLED)
    #                 if id == 8:
    #                     cx2, cy2 = cx, cy
    #                     cv2.circle(img, (cx2, cy2), 25, (255, 0, 255), cv2.FILLED)
    #                 dis = (math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2))
    #                 if draw:
    #                     cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    #     return dis
    def findPosition(self, img, draw=True):
    
        Plist = []
        
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handlms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    Plist.append([id,cx,cy])
    
        return Plist


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volbar = 400
    volper = 0

    volMin, volMax = volume.GetVolumeRange()[:2]
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
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

        #draw lines 
        # cv2.line(img, (5, 90), (25, 90), (0, 22, 250), 1)
        # cv2.line(img, (5, 90), (5, 220), (0, 22, 250), 1)
        # cv2.rectangle(img, (15, 200 - int(dis * 100 / 350)), (15, 210), (0, 22, 250), thick)
        # cv2.line(img, (25, 90), (25, 220), (0, 22, 250), 1)
        # cv2.line(img, (5, 220), (25, 220), (0, 22, 250), 1)
        # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (98, 33, 4), 3)
        # cv2.putText(img, str(int(dis * 100 / 350)) + '%', (10, 260), cv2.FONT_HERSHEY_PLAIN, 3, (0, 22, 250), 3)
        
            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255),4)
            cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()