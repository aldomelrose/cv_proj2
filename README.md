# My New Project

## TLDR if you want to see the models at work, just clone the whole repo and run models_inference.py

## Overview
This project focuses on implementing and using classical machine learning models for various tasks. The workflow includes data preprocessing, training, and inference, with a clear structure for managing code, data, and results.

---

## Folder Structure

### `classicalML`
This folder contains the core scripts for model training, inference, and data preprocessing:
- **`models_inference.py`**  
  Runs inference for XIB, Random Forest, and SVM models. This script acts as the main application for model predictions.
  
- **`models_train.py`**  
  Trains the models and exports them as `.joblib` files. These files are serialized versions of the trained models, ready for deployment or inference.

- **`preprocessing.py`**  
  Processes the data using MediaPipe to extract landmarks and saves them as `.npy` files in the `processed_data` folder.

---

### `data`
This folder manages raw and augmented datasets:
- **Raw Data**: Used for collecting the initial dataset.
- **Augmented Data**: Contains variations of the dataset generated to improve model generalization.

---

### `processed_data`
This folder holds preprocessed data that has already been passed through MediaPipe. The landmarks from the dataset images are stored as `.npy` files, ready for training or inference.

---

### `results`
Contains the results of model training and associated artifacts:
- **Date-Time Stamps**: Each folder is named with a timestamp indicating the training date and time.
- **Contents**:
  - `.joblib` files: Serialized versions of the trained models. These can be plugged into the inference script for predictions.
  - `training_report.pdf`: A summarized report of the training process and results for the corresponding timestamp.

---

## Usage
1. **Data Preprocessing**:  
   Use `preprocessing.py` to extract landmarks from raw image datasets and save them in `processed_data`.

2. **Model Training**:  
   Run `models_train.py` to train the models and export the `.joblib` files.

3. **Inference**:  
   Use `models_inference.py` to make predictions using trained models.

