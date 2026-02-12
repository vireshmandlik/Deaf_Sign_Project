import cv2
import mediapipe as mp
import numpy as np
import csv
import time

from mediapipe.tasks import python
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions
)

# -------------------------------
# Setup Hand Detector
# -------------------------------
options = HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# -------------------------------
# Open Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

gesture_name = input("Enter Gesture Name (HELLO / YES / NO): ")

print("Press 's' to save sample")
print("Press 'q' to quit")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = frame[:, :, ::-1]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int((time.time() - start_time) * 1000)

    result = detector.detect_for_video(mp_image, timestamp)

    if result and result.hand_landmarks:

        hand = result.hand_landmarks[0]
        landmark_list = []

        for lm in hand:
            landmark_list.append(lm.x)
            landmark_list.append(lm.y)
            landmark_list.append(lm.z)

            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        key = cv2.waitKey(1)

        # Save sample
        if key == ord('s'):

            with open("gesture_dataset.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(landmark_list + [gesture_name])

            print("✅ Sample Saved")

    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Dataset Collector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
