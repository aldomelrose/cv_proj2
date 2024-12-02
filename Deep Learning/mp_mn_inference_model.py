import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import queue
import time


class SignLanguageRecognizer:
    def __init__(self, model_path):
        # Initialize Keras model
        self.model = load_model(model_path)
        print("Model loaded successfully!")

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Define class names
        self.class_names = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
        ]

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Sign Language Recognition")

        # Create GUI elements
        self.frame_label = ttk.Label(self.root)
        self.frame_label.pack()

        self.prediction_label = ttk.Label(self.root,
                                          text="Prediction: None",
                                          font=('Arial', 14))
        self.prediction_label.pack()

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam")

        # Set up processing variables
        self.running = True
        self.last_prediction = None
        self.prediction_confidence = 0.0
        self.prediction_smoothing = 0.7  # Smoothing factor for predictions

    def get_hand_bbox(self, frame):
        """Detect hand and return bounding box coordinates"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Get bounding box
                h, w, _ = frame.shape
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

                # Add padding
                padding = 20
                x_min = max(0, int(x_min - padding))
                y_min = max(0, int(y_min - padding))
                x_max = min(w, int(x_max + padding))
                y_max = min(h, int(y_max + padding))

                return (x_min, y_min, x_max, y_max)

        return None

    def preprocess_image(self, img):
        """Preprocess image for model input"""
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def update_frame(self):
        """Update frame in GUI with stable processing"""
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                # Flip the frame for mirror effect
                frame = cv2.flip(frame, 1)

                # Process hand detection
                bbox = self.get_hand_bbox(frame)

                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    hand_crop = frame[y_min:y_max, x_min:x_max]

                    if hand_crop.size > 0:
                        processed_frame = self.preprocess_image(hand_crop)
                        predictions = self.model.predict(processed_frame, verbose=0)
                        predicted_class = self.class_names[np.argmax(predictions[0])]
                        confidence = float(np.max(predictions[0]))

                        # Smooth predictions
                        if self.last_prediction is None:
                            self.last_prediction = predicted_class
                            self.prediction_confidence = confidence
                        else:
                            if confidence > 0.5:  # Only update if confidence is high enough
                                if predicted_class == self.last_prediction:
                                    self.prediction_confidence = (
                                                self.prediction_smoothing * self.prediction_confidence +
                                                (1 - self.prediction_smoothing) * confidence)
                                else:
                                    self.last_prediction = predicted_class
                                    self.prediction_confidence = confidence

                        # Draw bounding box and prediction
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame,
                                    f"{self.last_prediction} ({self.prediction_confidence:.2f})",
                                    (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2)

                        # Update prediction label
                        self.prediction_label.configure(
                            text=f"Prediction: {self.last_prediction} ({self.prediction_confidence:.2f})"
                        )
                else:
                    # Clear prediction if no hand is detected
                    self.last_prediction = None
                    self.prediction_confidence = 0.0
                    self.prediction_label.configure(text="Prediction: None")

                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = ImageTk.PhotoImage(image=img)

                self.frame_label.img = img
                self.frame_label.configure(image=img)

            # Schedule next update
            self.root.after(10, self.update_frame)

    def run(self):
        """Main run method"""
        try:
            # Start frame update
            self.update_frame()

            # Start GUI main loop
            self.root.mainloop()

        finally:
            # Cleanup
            self.running = False
            self.cap.release()
            self.hands.close()


def main():
    try:
        # Update this path to match your .h5 model file
        model_path = "C:\ComVisProject\\final-project-inference-model\model\mediapipe_mobilenet.h5"

        recognizer = SignLanguageRecognizer(model_path)
        recognizer.run()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()