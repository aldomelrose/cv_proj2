import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import queue
import time


class SignLanguageRecognizer:
    def __init__(self, model_path, class_names_path):
        # Initialize TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load class names
        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

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

        # Set up frame processing queue
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True

    def preprocess_image(self, img):
        """Preprocess image for model input"""
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def process_frame(self, frame):
        """Process frame and return prediction"""
        processed_frame = self.preprocess_image(frame)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_frame)

        # Run inference
        self.interpreter.invoke()

        # Get prediction
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_class = self.class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        return predicted_class, confidence

    def update_frame(self):
        """Update frame in GUI"""
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                # Put frame in queue for processing
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

                # Convert frame for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=img)

                self.frame_label.img = img
                self.frame_label.configure(image=img)

            self.root.after(10, self.update_frame)

    def process_frames(self):
        """Process frames from queue"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                predicted_class, confidence = self.process_frame(frame)
                self.prediction_label.configure(
                    text=f"Prediction: {predicted_class} ({confidence:.2f})"
                )
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

    def run(self):
        """Main run method"""
        try:
            # Start frame processing thread
            processing_thread = threading.Thread(target=self.process_frames)
            processing_thread.start()

            # Start frame update
            self.update_frame()

            # Start GUI main loop
            self.root.mainloop()

        finally:
            # Cleanup
            self.running = False
            if processing_thread.is_alive():
                processing_thread.join()
            self.cap.release()


def main():
    try:
        # Update these paths to match your model and class names files
        model_path = 'C:\ComVisProject\\final-project-inference-model\model\MobileNet.tflite'
        class_names_path = 'C:\ComVisProject\\final-project-inference-model\model\class_names.txt'

        recognizer = SignLanguageRecognizer(model_path, class_names_path)
        recognizer.run()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()