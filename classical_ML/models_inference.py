import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import joblib

class SignLanguageInference:
    def __init__(self, model_path, confidence_threshold=0.7, is_xgboost=False):
        # Load the model
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
        
        self.is_xgboost = is_xgboost
        # Create simple label mapping A-Y (excluding J)
        self.label_mapping = {i: chr(65 + i) for i in range(24)}
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3
        )
        
        self.prediction_history = deque(maxlen=5)
        
        print("Model loaded successfully!")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Using XGBoost: {self.is_xgboost}")

    def get_letter(self, prediction):
        """Convert numeric prediction to letter if using XGBoost."""
        if self.is_xgboost:
            return self.label_mapping[prediction]
        return prediction

    def extract_features(self, frame):
        """Extract hand landmarks from frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None, None
        
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract features
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(features), hand_landmarks

    def draw_prediction(self, frame, hand_landmarks, prediction, confidence):
        """Draw landmarks and prediction with confidence."""
        # Draw hand landmarks
        self.mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Draw bounding box and prediction
        h, w, _ = frame.shape
        
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        x1 = int(min(x_coords) * w) - 10
        y1 = int(min(y_coords) * h) - 10
        x2 = int(max(x_coords) * w) + 10
        y2 = int(max(y_coords) * h) + 10
        
        # Color based on confidence (green to red)
        color = (0, int(255 * confidence), int(255 * (1 - confidence)))
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw prediction and confidence
        text = f"{prediction} ({confidence:.2f})"
        cv2.putText(frame, 
                   text,
                   (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, 
                   color, 
                   2)
        
        return frame

    def draw_instructions(self, frame):
        """Draw instructions on frame."""
        h, w = frame.shape[:2]
        instructions = [
            "Q: Quit",
            "C: Clear history"
        ]
        
        y = h - 20 * len(instructions)
        for instruction in instructions:
            cv2.putText(frame, instruction, (w-150, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
        return frame

    def run_inference(self, camera_id=1):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
            
        fps = cv2.getTickFrequency()
        
        print("\nStarting inference...")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 'c' to clear prediction history")
        
        while cap.isOpened():
            t1 = cv2.getTickCount()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract features
            features, hand_landmarks = self.extract_features(frame)
            
            # Always draw landmarks if detected
            if hand_landmarks is not None:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Make predictions if features are available
            if features is not None:
                prediction = self.model.predict([features])[0]
                probabilities = self.model.predict_proba([features])[0]
                confidence = np.max(probabilities)
                
                # Convert numeric prediction to letter if using XGBoost
                prediction = self.get_letter(prediction)
                
                if confidence >= self.confidence_threshold:
                    self.prediction_history.append((prediction, confidence))
                    
                    if len(self.prediction_history) >= 3:
                        prediction_weights = {}
                        for pred, conf in self.prediction_history:
                            prediction_weights[pred] = prediction_weights.get(pred, 0) + conf
                        
                        final_prediction = max(prediction_weights, key=prediction_weights.get)
                        avg_confidence = prediction_weights[final_prediction] / len(self.prediction_history)
                        
                        # Draw only the box and prediction text
                        h, w, _ = frame.shape
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        
                        x1 = int(min(x_coords) * w) - 10
                        y1 = int(min(y_coords) * h) - 10
                        x2 = int(max(x_coords) * w) + 10
                        y2 = int(max(y_coords) * h) + 10
                        
                        # Color based on confidence
                        color = (0, int(255 * avg_confidence), int(255 * (1 - avg_confidence)))
                        
                        # Draw box and prediction
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        text = f"{final_prediction} ({avg_confidence:.2f})"
                        cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Calculate and display FPS
            t2 = cv2.getTickCount()
            time_taken = (t2 - t1) / fps
            fps_text = f"FPS: {1.0 / time_taken:.2f}"
            cv2.putText(frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add instructions
            frame = self.draw_instructions(frame)
            
            # Show frame
            cv2.imshow('Sign Language Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.prediction_history.clear()
                print("Prediction history cleared")
        
        cap.release()
        cv2.destroyAllWindows()

    def run_inference_on_frame(self, frame):
        features, hand_landmarks = self.extract_features(frame)
        final_prediction = None
        avg_confidence = 0

        if hand_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if features is not None:
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            confidence = np.max(probabilities)
            
            prediction = self.get_letter(prediction)
            
            if confidence >= self.confidence_threshold:
                self.prediction_history.append((prediction, confidence))
                
                if len(self.prediction_history) >= 3:
                    prediction_weights = {}
                    for pred, conf in self.prediction_history:
                        prediction_weights[pred] = prediction_weights.get(pred, 0) + conf
                    
                    final_prediction = max(prediction_weights, key=prediction_weights.get)
                    avg_confidence = prediction_weights[final_prediction] / len(self.prediction_history)
                    
                    # Draw only the box and prediction text
                    h, w, _ = frame.shape
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    
                    x1 = int(min(x_coords) * w) - 10
                    y1 = int(min(y_coords) * h) - 10
                    x2 = int(max(x_coords) * w) + 10
                    y2 = int(max(y_coords) * h) + 10
                    
                    # Color based on confidence
                    color = (0, int(255 * avg_confidence), int(255 * (1 - avg_confidence)))
                    
                    # Draw box and prediction
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{final_prediction} ({avg_confidence:.2f})"
                    cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
        
        return frame, final_prediction, avg_confidence

def main():
    model_path = 'results/trial1_20241112_220259/svm_model.joblib'     # change to your model path 
    
    try:
        # Set is_xgboost=True when using XGBoost model
        inferencer = SignLanguageInference(model_path, is_xgboost=False)
        inferencer.run_inference()
    except FileNotFoundError:
        print("\nError: Model file not found!")
        print("Please check the path and make sure you have trained the model.")
        print(f"Looking for model at: {model_path}")

if __name__ == "__main__":
    main()