import os
import cv2
import time

# Directory to store images
DATA_DIR = '/Users/aldridgemelrose/Documents/SIT/2.1/Computer_Vision/team_project/sign_language_project/Yhands_dataset_cleaned'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_next_image_number(folder_path):
    existing_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    if not existing_images:
        return 0
    # Extract numbers from filenames like 'A_023.jpg'
    numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_images]
    return max(numbers) + 1 if numbers else 0

def main():
    # Get folder name from user
    folder_name = input("Enter the folder name (e.g., 'A', 'B', etc.): ").upper()
    folder_path = os.path.join(DATA_DIR, folder_name)
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        start_num = 0
    else:
        start_num = get_next_image_number(folder_path)
        print(f"Found existing images. Starting from number {start_num}")
    
    # Initialize camera
    cap = cv2.VideoCapture(1)
    
    image_count = start_num
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Create display frame with guidelines
        display_frame = frame.copy()
        cv2.putText(display_frame, f'Folder: {folder_name}, Next Image: {folder_name}_{image_count:03d}.jpg',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, 'Press SPACE to capture, ESC to exit, R to redo last',
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw center guide
        h, w = frame.shape[:2]
        cv2.rectangle(display_frame, (w//2-100, h//2-100), (w//2+100, h//2+100), (0, 255, 0), 2)
        
        cv2.imshow('Sign Language Collector', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # SPACE key
            # Save image
            image_path = os.path.join(folder_path, f'{folder_name}_{image_count:03d}.jpg')
            cv2.imwrite(image_path, frame)
            print(f"Saved {image_path}")
            image_count += 1
            
        elif key == ord('r') and image_count > start_num:  # R key - redo last image
            last_image = os.path.join(folder_path, f'{folder_name}_{image_count-1:03d}.jpg')
            if os.path.exists(last_image):
                os.remove(last_image)
                image_count -= 1
                print(f"Deleted last image. Ready to recapture {folder_name}_{image_count:03d}.jpg")
        
        elif key == 27:  # ESC key
            print("\nExiting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()