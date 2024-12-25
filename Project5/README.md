# Breast Cancer Image Classification using CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) for classifying breast cancer Fine Needle Aspiration (FNA) images as either benign or malignant.

## Dataset
- Source: Dataset2/FNA
- Total Images: 1724 files belonging to 2 classes
- Split: 
  - Training: 1380 files (80%)
  - Validation: 344 files (20%)
- Image Size: 224x224 pixels, 3 channels (RGB)

## Dependencies
- TensorFlow
- Matplotlib
- Python 3.x

## Model Architecture
The CNN model consists of the following layers:

1. Input Layer (224x224x3)

2. Convolutional Blocks:
   - First Block:
     - Conv2D (32 filters, 3x3 kernel, ReLU activation)
     - MaxPooling2D
     - BatchNormalization
   
   - Second Block:
     - Conv2D (64 filters, 3x3 kernel, ReLU activation)
     - MaxPooling2D
     - BatchNormalization
   
   - Third Block:
     - Conv2D (128 filters, 3x3 kernel, ReLU activation)
     - MaxPooling2D
     - BatchNormalization

3. Dense Layers:
   - Flatten
   - Dense (128 units, ReLU activation)
   - Dropout (0.5)
   - Dense (1 unit, Sigmoid activation)

## Data Preprocessing
1. Image Augmentation:
   - Random horizontal flip
   - Random rotation (0.2)
   - Random zoom (0.1)
2. Pixel Normalization (1/255)

## Training Details
- Optimizer: Adam (learning rate = 0.001)
- Loss Function: Binary Cross-entropy
- Metrics: Accuracy
- Epochs: 20
- Batch Size: 32

## Model Performance
Final Metrics:
- Training Accuracy: 0.8703
- Validation Accuracy: 0.9070
- Training Loss: 0.3531
- Validation Loss: 0.2968

## Test Results
The model was tested on 14 unseen images with the following distribution:
- 9 Malignant predictions
- 5 Benign predictions
- Confidence scores ranging from 0.5526 to 0.9999

## Files in the Project
1. Jupyter Notebook (22CS10048_P5.ipynb)
2. Saved Model (breast_cancer_model.h5)
3. Dataset Directory (Dataset2/FNA)
4. Test Images Directory (Dataset2/test)

## Usage
1. Install required dependencies
```python
import tensorflow as tf
import matplotlib.pyplot as plt

train_dataset = tf.keras.utils.image_dataset_from_directory(
    'Dataset2/FNA',
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    verbose=1
)

predictions = model.predict(test_dataset)