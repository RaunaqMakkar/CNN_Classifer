# cnn_image_classifier.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ==== Configuration ====
MODEL_PATH    = "image_classifier.keras"
IMAGE_PATH    = "download.jpg"   # ←←← Put your image filename (with extension) here
EPOCHS        = 30           # you can increase for better accuracy
BATCH_SIZE    = 64

# ==== 1. Load & Preprocess CIFAR-10 ====
print("[INFO] Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize

class_names = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

# ==== 2. Build or Load Model ====
if os.path.exists(MODEL_PATH):
    print(f"[INFO] Found saved model '{MODEL_PATH}'. Loading...")
    model = keras.models.load_model(MODEL_PATH)
else:
    print("[INFO] No saved model—building a new one...")

    # Optional data‐augmentation stage
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])

    model = models.Sequential([
        data_augmentation,  # comment out if you don’t want augmentation
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128,(3,3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # softmax for proper probabilities
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"[INFO] Training for {EPOCHS} epochs...")
    model.fit(x_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(x_test, y_test))
    print(f"[INFO] Saving model to '{MODEL_PATH}'")
    model.save(MODEL_PATH)

# ==== 3. Evaluate on Test Data ====
print("[INFO] Evaluating on CIFAR-10 test set...")
loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Loss: {loss:.4f}   Test Accuracy: {test_accuracy:.4f}")

# ==== 4. Classify an External Image ====
print(f"[INFO] Loading external image '{IMAGE_PATH}'...")
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not find image at path: {IMAGE_PATH}")

# Preprocess to (32,32,3) RGB
img = cv2.resize(img, (32,32))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype("float32") / 255.0
input_tensor = np.expand_dims(img, axis=0)  # shape: (1,32,32,3)

# Predict
preds = model.predict(input_tensor)
i = np.argmax(preds[0])
print(f"✅ Predicted class: {class_names[i]}  (confidence: {preds[0][i]:.4f})")

# Show the image and label
plt.imshow(img)
plt.title(f"Prediction: {class_names[i]}\nModel Accuracy: {test_accuracy:.2%} | Confidence: {preds[0][i]:.2%}")
plt.axis('off')
plt.show()
