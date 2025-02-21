# multi-class-image-classification
# Rice Variety Classification Using Deep Learning & Big Data

## Overview
This project focuses on classifying rice varieties using a Convolutional Neural Network (CNN) model trained on a dataset of 75,000 rice images spanning five different varieties. The model achieves a high classification accuracy of **99.10%** on the test set.

## Key Features
* **Deep Learning Model:** Built a CNN for multi-class classification of rice images
* **Dataset Processing:** Preprocessed and augmented images to enhance model generalization
* **Big Data Integration:** Utilized Apache Spark for efficient data handling and parallel computation 
* **Scalability:** Distributed processing using multiple Linux virtual machines (VMs) to optimize performance
* **Performance Metrics:** Achieved high sensitivity, specificity, and F1-score

## Dataset Preparation
The dataset consists of 75,000 images of five rice varieties. Preprocessing steps included:
* Resizing images to **150×150 pixels**
* Normalizing pixel values
* Data augmentation techniques:
  * Rotation
  * Width & height shift
  * Shear transformation
  * Zoom
  * Horizontal flips

## Model Architecture
The CNN architecture consists of:

### Feature Extraction Layers
* Convolutional layers: **32, 64 filters** (ReLU activation)
* MaxPooling layers for downsampling

### Fully Connected Layers  
* Dense layer: **128 neurons**
* Output layer: Softmax activation for classification

## Training Process
* **Optimizer:** Adam (learning rate = **0.001**)
* **Loss Function:** Categorical Cross-Entropy
* **Batch Size:** 32
* **Epochs:** 10

## Big Data Implementation
Apache Spark was used to handle large-scale datasets efficiently:
* **Distributed Training:** Leveraged **10 Linux VMs** to speed up processing
* **Scalability Testing:** Benchmarked Spark performance using **2 vs. 7 vs. 10VMs**, showing improved efficiency with increased resources
* **Processing Time:** **59 minutes** using 10 VMs

## Model Performance
Final evaluation metrics:
* **Training Accuracy:** **98.25%**
* **Validation Accuracy:** **99.00%**
* **Test Accuracy:** **99.10%**
* **Loss Reduction:**
  * Training Loss: **0.0534** → **0.0279** (Validation Loss)
* **Performance Metrics:**
  * **Sensitivity (Recall):** **99.14%**
  * **Specificity:** **99.79%**
  * **F1-score:** **99.14%**
