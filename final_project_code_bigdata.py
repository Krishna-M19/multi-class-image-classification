# -*- coding: utf-8 -*-

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Rice Image Classification with Spark") \
    .getOrCreate()

# Dataset path
dataset_path = "/home/sat3812/Downloads/Rice_Image_Dataset"

# Create Spark DataFrame for images and labels
data = []
classes = os.listdir(dataset_path)
for rice_class in classes:
    class_path = os.path.join(dataset_path, rice_class)
    if os.path.isdir(class_path):
        for image in os.listdir(class_path):
            data.append((os.path.join(class_path, image), rice_class))

df_spark = spark.createDataFrame(data, ["image", "label"])

# Convert Spark DataFrame to Pandas DataFrame for visualization and preprocessing
df = df_spark.toPandas()

# Display classes and sample images
print("Classes in dataset:", df['label'].unique())
for label in df['label'].unique():
    print(f"Sample images in {label}:")
    print(df[df['label'] == label].iloc[:5]['image'].values)

# Plot class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=df['label'])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.show()

# Visualize sample images for each class
num_categories = len(df['label'].unique())
fig = plt.figure(figsize=(20, num_categories * 5))
gs = GridSpec(num_categories, 4, figure=fig)

for i, category in enumerate(df['label'].unique()):
    subset = df[df['label'] == category].iloc[:4]
    filepaths = subset['image'].values
    filenames = [filepath.split("/")[-1] for filepath in filepaths]
    for j, (filepath, filename) in enumerate(zip(filepaths, filenames)):
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(plt.imread(filepath))
        ax.axis('off')
        ax.set_title(filename, fontsize=8, color='black', pad=5)
        if j == 0:
            ax.text(-0.3, 0.5, category, fontsize=12, color='darkblue', weight='bold',
                    transform=ax.transAxes, va='center', ha='center', rotation=90)
plt.subplots_adjust(hspace=0.5, wspace=0.2)
plt.show()

# Split data into train and test sets
train_df, test_df = df_spark.randomSplit([0.8, 0.2], seed=42)
train_df = train_df.toPandas()
test_df = test_df.toPandas()

# Encode labels
encoder = LabelEncoder()
train_df['encoded_label'] = encoder.fit_transform(train_df['label'])
test_df['encoded_label'] = encoder.transform(test_df['label'])

# Data augmentation
image_size = (50, 50)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
).repeat()
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Build the model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Set the input shape for the model
input_shape = (50, 50, 3)

# Create a Sequential model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data
history = model.fit_generator(train_generator,epochs=5,validation_data=test_generator)

# Visualize training progress
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate confusion matrix and classification report
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
class_labels = list(test_generator.class_indices.keys())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

report = classification_report(y_true, y_pred_classes, target_names=class_labels)
print("Classification Report:")
print(report)

# Save the model
model.save("rice_classifier_model_spark.h5")
print("Model saved as rice_classifier_model_spark.h5")
