import cv2
import mediapipe as mp
import numpy as np
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions
)

# -------------------------------
# Angle Calculation Function
# -------------------------------
def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    cosine_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (
        math.sqrt(ba[0]**2 + ba[1]**2) *
        math.sqrt(bc[0]**2 + bc[1]**2)
    )

    angle = math.degrees(math.acos(cosine_angle))
    return angle


# -------------------------------
# Loading Screen
# -------------------------------
loading_img = 255 * np.ones((250, 250, 3), np.uint8)
cv2.putText(loading_img, "Camera loading...", (50, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.namedWindow("Finger Counter", cv2.WINDOW_NORMAL)
cv2.imshow("Finger Counter", loading_img)
cv2.waitKey(1)


# -------------------------------
# Initialize Hand Detector
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

if not cap.isOpened():
    print("Camera not detected")
    exit()

# Optional resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

start_time = time.time()

print("Press 'q' on window to exit")


# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = frame[:, :, ::-1]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int((time.time() - start_time) * 1000)

    result = detector.detect_for_video(mp_image, timestamp)

    finger_count = 0

    # -------------------------------
    # Finger Detection
    # -------------------------------
    if result and result.hand_landmarks:

        hand = result.hand_landmarks[0]
        lm_list = []

        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((cx, cy))
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        if len(lm_list) >= 21:

            # Thumb
            thumb_angle = calculate_angle(
                lm_list[2], lm_list[3], lm_list[4]
            )

            if thumb_angle > 150:
                finger_count += 1

            # Other Fingers
            finger_angles = [
                (5, 6, 8),    # Index
                (9, 10, 12),  # Middle
                (13, 14, 16), # Ring
                (17, 18, 20)  # Little
            ]

            for a, b, c in finger_angles:
                angle = calculate_angle(
                    lm_list[a],
                    lm_list[b],
                    lm_list[c]
                )

                if angle > 160:
                    finger_count += 1

    # -------------------------------
    # Display Output
    # -------------------------------
    cv2.putText(frame, f"Fingers: {finger_count}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 0, 0), 3)

    cv2.putText(frame, "Camera ON",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("Finger Counter", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break


# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
