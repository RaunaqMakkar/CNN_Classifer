# 🖼️ CNN Image Classifier

A deep learning project using Convolutional Neural Networks (CNN) to classify images from the CIFAR-10 dataset. This project demonstrates building, training, and deploying a CNN model for object recognition.

## 📋 Description

**CNN Image Classifier** is a TensorFlow/Keras-based application that:
- Trains a CNN model on the CIFAR-10 dataset (10 object classes)
- Automatically loads saved models or trains a new one
- Classifies external images with confidence scores
- Displays predictions with model accuracy and confidence metrics
- Includes data augmentation for improved training

## 🎯 Features

- **Data Augmentation**: Random horizontal flips and rotations to improve model robustness
- **Multi-Layer CNN**: 3 convolutional layers with max pooling for feature extraction
- **Dropout Regularization**: 50% dropout to prevent overfitting
- **Model Persistence**: Automatically saves/loads trained models
- **Confidence Scoring**: High-precision confidence predictions (6 decimal places)
- **Visual Output**: Displays predictions with model accuracy and confidence on the image

## 🏗️ CNN Architecture

```
Input (32×32×3)
    ↓
Data Augmentation (RandomFlip, RandomRotation)
    ↓
Conv2D (32 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (128 filters, 3×3) + ReLU
    ↓
Flatten
    ↓
Dropout (0.5)
    ↓
Dense (128 units) + ReLU
    ↓
Dense (10 units) + Softmax
    ↓
Output (10 classes)
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Dependencies
```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

## 🚀 Usage

1. **Place your image** in the project directory and update `IMAGE_PATH` in the script:
   ```python
   IMAGE_PATH = "your_image.jpg"
   ```

2. **Run the classifier**:
   ```bash
   python cnn_image_classifier.py
   ```

3. **View Results**:
   - Terminal output shows: Predicted class and confidence score
   - Plot displays: Image with prediction, model accuracy, and confidence

## 🏷️ CIFAR-10 Classes

- plane, car, bird, cat, deer, dog, frog, horse, ship, truck

## ⚙️ Configuration

Edit these parameters in the script:

```python
MODEL_PATH    = "image_classifier.keras"  # Model save location
IMAGE_PATH    = "add_your_image_path"     # Image to classify
EPOCHS        = 30                        # Training epochs
BATCH_SIZE    = 64                        # Batch size
```

**Plot:**
- Shows the input image with title containing prediction, model accuracy, and confidence percentage

## 🎓 Model Performance

- **Test Accuracy**: ~82% on CIFAR-10 test set
- **Training Time**: ~10-15 minutes (30 epochs on CPU)
- **Input Size**: 32×32 RGB images (auto-resized if different)

## 📝 Notes

- Model is automatically saved after training to avoid retraining
- Delete `image_classifier.keras` to retrain from scratch
- Input images are automatically resized to 32×32
- Images are normalized to [0, 1] range before prediction

## 🔧 Tips

- Increase `EPOCHS` for better accuracy (may take longer)
- Increase `BATCH_SIZE` for faster training on GPU
- Comment out data augmentation if not needed
- Use high-resolution images for better preprocessing

 **Framework**: TensorFlow/Keras | **Dataset**: CIFAR-10
