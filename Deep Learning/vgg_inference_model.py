import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import time

class RealtimeSignLanguage:
    def __init__(self, model_path):
        try:
            # Load model info
            self.model_info = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Debug print to check model_info contents
            print("Loaded model info keys:", self.model_info.keys())
            
            # Try to get num_classes, with fallback
            if 'num_classes' not in self.model_info:
                # Check if we can infer from the model state dict
                if 'model_state_dict' in self.model_info:
                    # Look for the final layer weights to determine number of classes
                    final_layer_weight = self.model_info['model_state_dict']['classifier.6.weight']
                    num_classes = final_layer_weight.size(0)
                    print(f"Inferred num_classes from model weights: {num_classes}")
                else:
                    # Fallback to a default value if necessary
                    num_classes = 24  # Common for ASL alphabet
                    print(f"Using default num_classes: {num_classes}")
            else:
                num_classes = self.model_info['num_classes']
            
            # Create model architecture
            self.model = models.vgg16(pretrained=False)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, num_classes)
            
            # Load trained weights
            if 'model_state_dict' not in self.model_info:
                raise KeyError("model_state_dict not found in saved model file")
            
            self.model.load_state_dict(self.model_info['model_state_dict'])
            self.model.eval()
            
            # Get class mapping with fallback
            if 'idx_to_class' not in self.model_info:
                # Create default mapping for ASL alphabet
                self.idx_to_class = {i: chr(65 + i) for i in range(num_classes)}
                print("Using default alphabet mapping")
            else:
                self.idx_to_class = self.model_info['idx_to_class']
            
            # Define image transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            
            # Initialize webcam
            self.cap = cv2.VideoCapture(0)
            
            # Set region of interest (ROI) for hand signs
            self.roi_top = 100
            self.roi_bottom = 400
            self.roi_left = 320
            self.roi_right = 620
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            print(f"Model info keys available: {self.model_info.keys() if hasattr(self, 'model_info') else 'No model info loaded'}")
            raise
    
    def preprocess_frame(self, frame):
        """Preprocess webcam frame for inference"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply transformations
        img_tensor = self.transform(pil_image)
        return img_tensor.unsqueeze(0)
    
    def predict_frame(self, frame):
        """Predict from preprocessed frame"""
        with torch.no_grad():
            outputs = self.model(frame)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            return self.idx_to_class[prediction.item()], confidence.item()
    
    def run(self):
        print("Starting webcam feed...")
        print("Press 'q' to quit")
        print("Place your hand sign in the green box")
        
        # For FPS calculation
        prev_time = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Draw ROI for hand placement
            cv2.rectangle(frame, 
                        (self.roi_left, self.roi_top), 
                        (self.roi_right, self.roi_bottom), 
                        (0, 255, 0), 2)
            
            # Extract ROI
            roi = frame[self.roi_top:self.roi_bottom, 
                       self.roi_left:self.roi_right]
            
            # Preprocess and predict
            processed_frame = self.preprocess_frame(roi)
            prediction, confidence = self.predict_frame(processed_frame)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Display prediction and FPS
            cv2.putText(frame, f"Sign: {prediction}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {confidence:.2%}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.2f}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Show frame
            cv2.imshow('Sign Language Recognition', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

def main():
    # Initialize model
    model_path = 'C:\ComVisProject\\final-project-inference-model\model\\vgg_pretrain.pth'  # Update with your model path
    
    try:
        sign_language = RealtimeSignLanguage(model_path)
        sign_language.run()
    except KeyboardInterrupt:
        print("\nStopping webcam feed...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()