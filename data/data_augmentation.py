import os
import cv2
import numpy as np
from albumentations import (
    Compose, RandomBrightnessContrast, GaussianBlur, 
    RandomRotate90, Rotate, Flip, Affine,
    MedianBlur, GaussNoise, RandomScale,
    SafeRotate, RandomShadow, ColorJitter
)

def create_augmentation_pipeline():
    return Compose([
        # Light adjustments (preserve hand details)
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        
        # Geometric transformations (maintain gesture recognition)
        SafeRotate(limit=15, p=0.7),  # Small rotations
        RandomScale(scale_limit=0.15, p=0.5),  # Subtle scaling
        Affine(
            scale=(0.8, 1.2),
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            rotate=(-10, 10),
            p=0.7
        ),
        
        # Noise and blur (simulate different camera conditions)
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        GaussianBlur(blur_limit=(3, 5), p=0.3),
        MedianBlur(blur_limit=3, p=0.3),
        
        # Lighting effects
        RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
    ])

def augment_dataset(input_dir, output_dir, augmentations_per_image=9):
    """
    Augment images in the input directory and save to output directory.
    
    Args:
        input_dir: Path to input directory containing letter folders
        output_dir: Path to output directory
        augmentations_per_image: Number of augmented versions to create per original image
    """
    transform = create_augmentation_pipeline()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each letter folder
    for letter in os.listdir(input_dir):
        letter_path = os.path.join(input_dir, letter)
        if not os.path.isdir(letter_path):
            continue
            
        # Create output letter folder
        output_letter_path = os.path.join(output_dir, letter)
        os.makedirs(output_letter_path, exist_ok=True)
        
        # Process each image in the letter folder
        for img_name in os.listdir(letter_path):
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            # Read image
            img_path = os.path.join(letter_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
                
            # Generate augmented versions
            for i in range(augmentations_per_image):
                # Apply transformations
                augmented = transform(image=image)
                aug_image = augmented['image']
                
                # Generate output filename
                base_name = os.path.splitext(img_name)[0]
                aug_name = f"{base_name}_aug_{i+1}.jpg"
                output_path = os.path.join(output_letter_path, aug_name)
                
                # Save augmented image
                cv2.imwrite(output_path, aug_image)

def main():
    input_dir = "data/hands_dataset_cleaned"  # Your original dataset path
    output_dir = "data/hands_dataset_augmented"  # Path for augmented dataset
    augmentations_per_image = 9  # Will give you ~1000 images per class (100 * 10 total)
    
    augment_dataset(input_dir, output_dir, augmentations_per_image)
    print("Augmentation completed!")

if __name__ == "__main__":
    main()