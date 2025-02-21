# multi-class-image-classification
Rice Variety Classification Using Deep Learning &amp; Big Data
Developed and deployed a CNN model for multi-class classification of 75,000 rice images across five varieties, achieving 99.10% test accuracy.
Preprocessed and augmented the dataset, resizing images to 150×150 pixels, normalizing pixel values, and applying rotation, shift, shear, zoom, and horizontal flips to enhance model generalization.
Implemented a deep learning architecture with:
Convolutional Layers (Feature Extraction: 32, 64 filters, ReLU activation)
MaxPooling Layers (Downsampling for computational efficiency)
Fully Connected Layers (Dense: 128 neurons, Softmax for classification)
Optimized training using Adam optimizer (learning rate = 0.001) and Categorical Cross-Entropy loss, training for 10 epochs with batch size = 32.
Utilized Apache Spark for Big Data handling, enabling efficient processing and parallel computation for large-scale datasets.
Improved processing speed with distributed computing, leveraging 7 Linux VMs to complete the task in 59 minutes, demonstrating efficient workload distribution.
Benchmarked Spark performance on 2 vs. 7 VMs, showcasing improved scalability and efficiency in distributed model training.
Evaluated model performance using a Confusion Matrix, Classification Report, and KPI tracking, analyzing misclassified images for further improvements.
Final Results
Training Accuracy: 98.25%, Validation Accuracy: 99.00%, Test Accuracy: 99.10%
Loss Reduction: Training Loss 0.0534 → 0.0279 Validation Loss
Model Performance Metrics:
Sensitivity (Recall): 99.14%
Specificity: 99.79%
F1-score: 99.14%
