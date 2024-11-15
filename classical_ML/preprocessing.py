import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
import joblib

class SignLanguagePreprocessor:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3
        )
    
    def extract_features(self, image):
        """Extract hand landmarks from a single image."""
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extract landmarks
        landmarks = results.multi_hand_landmarks[0]
        
        # Get features (x, y, z coordinates)
        features = []
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
            
        return np.array(features)
    
    def process_dataset(self, data_dir, verbose=True):
        """Process entire dataset and return features and labels."""
        features = []
        labels = []
        stats = {}
        
        # Get all gesture folders (A, B, C, etc.)
        gesture_folders = [f for f in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, f))]
        
        for gesture in sorted(gesture_folders):
            gesture_path = os.path.join(data_dir, gesture)
            
            if verbose:
                print(f"\nProcessing gesture {gesture}")
            
            # Get jpg files
            image_files = [f for f in os.listdir(gesture_path) 
                         if f.endswith('.jpg') and not f.startswith('.')]
            
            if verbose:
                print(f"Found {len(image_files)} images")
            
            successful = 0
            failed = 0
            
            # Process images
            for image_file in tqdm(image_files, desc=f"Processing gesture {gesture}", 
                                 disable=not verbose):
                image_path = os.path.join(gesture_path, image_file)
                
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        failed += 1
                        continue
                    
                    hand_features = self.extract_features(image)
                    if hand_features is not None:
                        features.append(hand_features)
                        labels.append(gesture)
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    if verbose:
                        print(f"Error processing {image_file}: {str(e)}")
                    failed += 1
            
            stats[gesture] = {
                'successful': successful,
                'failed': failed,
                'total': len(image_files)
            }
            
            if verbose:
                print(f"\nGesture {gesture} statistics:")
                print(f"Successful extractions: {successful}")
                print(f"Failed extractions: {failed}")
        
        if len(features) == 0:
            raise ValueError("No valid samples were processed. Check the dataset structure.")
        
        features = np.array(features)
        labels = np.array(labels)
        
        if verbose:
            self._print_dataset_statistics(features, labels)
        
        return features, labels, stats
    
    def _print_dataset_statistics(self, features, labels):
        """Print dataset statistics."""
        print("\nFinal dataset statistics:")
        print(f"Total samples: {len(features)}")
        print("\nSamples per class:")
        for label in sorted(np.unique(labels)):
            count = np.sum(labels == label)
            print(f"Class {label}: {count} samples")

def prepare_data(data_dir, test_size=0.2, random_state=42):
    """Prepare dataset for training and save to pickle files."""
    # Initialize preprocessor
    preprocessor = SignLanguagePreprocessor()
    
    # Process dataset
    print("Processing dataset...")
    X, y, stats = preprocessor.process_dataset(data_dir)
    print("\nDataset processed successfully!")
    print(f"Feature shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print("\nSplit created:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Create processed_data directory if it doesn't exist
    processed_dir = 'processed_data'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save the processed data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.save(os.path.join(processed_dir, f'X_train_{timestamp}.npy'), X_train)
    np.save(os.path.join(processed_dir, f'X_test_{timestamp}.npy'), X_test)
    np.save(os.path.join(processed_dir, f'y_train_{timestamp}.npy'), y_train)
    np.save(os.path.join(processed_dir, f'y_test_{timestamp}.npy'), y_test)
    
    # Save processing statistics
    joblib.dump(stats, os.path.join(processed_dir, f'processing_stats_{timestamp}.joblib'))
    
    print(f"\nProcessed data saved to {processed_dir}/")
    print(f"Timestamp: {timestamp}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Set data directory
    DATA_DIR = '/Users/aldridgemelrose/Documents/SIT/2.1/Computer_Vision/team_project/sign_language_project/data/hands_dataset_augmented'
    
    try:
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(DATA_DIR)
        print("Data preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")