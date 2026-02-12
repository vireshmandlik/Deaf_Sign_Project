import cv2
import numpy as np
import pickle
import time

from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

import mediapipe as mp


# -------------------------------
# Load ML Model
# -------------------------------
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Model Loaded")


# -------------------------------
# MediaPipe Tasks Setup
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
# Webcam Start
# -------------------------------
cap = cv2.VideoCapture(0)
start_time = time.time()

print("Press ESC to exit")


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

    # -------------------------------
    # Landmark Processing
    # -------------------------------
    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        landmark_list = []

        for lm in hand:
            landmark_list.append(lm.x)
            landmark_list.append(lm.y)
            landmark_list.append(lm.z)

        # Convert to numpy
        input_data = np.array(landmark_list).reshape(1, -1)

        # ML Prediction
        prediction = model.predict(input_data)
        confidence = model.predict_proba(input_data)

        gesture = prediction[0]
        conf = np.max(confidence)

        # Display Text
        cv2.putText(frame, f"{gesture} ({conf:.2f})",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
