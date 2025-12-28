# Dog Breed Classifier

A deep learning project to classify dog breeds using XceptionNet architecture. This model can identify dog breeds from images with high accuracy.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Scripts](#scripts)
- [Results](#results)

## ✨ Features

- **Multi-breed Classification**: Identifies 120+ dog breeds
- **Pre-trained XceptionNet Model**: Leverages transfer learning for better accuracy
- **Real-time Detection**: Live detection from webcam or image files
- **Data Pipeline**: Complete scripts for data processing, training, and evaluation
- **High Accuracy**: Model trained on comprehensive dog breed dataset

## 📁 Project Structure

```
Dog_breed_classifier/
├── Code.py                          # Main application script
├── breed_classifier_xception.h5     # Pre-trained model weights
├── requirements.txt                 # Python dependencies
├── scripts/
│   ├── train_model.py              # Model training script
│   ├── evaluate_model.py           # Model evaluation
│   ├── live_detect.py              # Real-time detection from webcam
│   ├── resize_images.py            # Image resizing utility
│   ├── normalize_images.py         # Image normalization
│   ├── split_data.py               # Train/test data splitting
│   ├── data_generator.py           # Data augmentation
│   └── check_model.py              # Model validation
├── image/                           # Sample dog breed images
├── annotation/                      # Breed annotation data
├── Dataset/                         # Training dataset
├── train_split/                     # Training split data
├── test_split/                      # Test split data
├── resized_images/                  # Processed images
└── README.md                        # This file
```

## 🔧 Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pillow

## 📥 Installation

1. **Clone the repository**
```bash
git clone https://github.com/ujjwalutkarsh21/Dog_breed_classifier.git
cd Dog_breed_classifier
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Using the Main Application
```bash
python Code.py
```

### Real-time Detection from Webcam
```bash
python scripts/live_detect.py
```

### Train the Model
```bash
python scripts/train_model.py
```

### Evaluate the Model
```bash
python scripts/evaluate_model.py
```

### Process Images
```bash
# Resize images
python scripts/resize_images.py

# Normalize images
python scripts/normalize_images.py

# Split data into train/test sets
python scripts/split_data.py
```

## 🧠 Model Details

- **Architecture**: XceptionNet (Extreme Inception)
- **Input Shape**: (299, 299, 3)
- **Transfer Learning**: Pre-trained on ImageNet
- **Number of Classes**: 120+ dog breeds
- **Output**: Breed classification with confidence scores

### Model Performance
The model achieves high accuracy on the dog breed classification task through:
- Deep convolutional neural networks
- Data augmentation
- Transfer learning from ImageNet
- Fine-tuning on dog breed dataset

## 📊 Dataset

The project uses a comprehensive dog breed dataset with:
- Multiple images per breed for robust training
- Organized by breed in annotation folders
- Preprocessed and resized for optimal performance
- Split into training and testing sets

### Dataset Structure
```
annotation/
├── n02085620-Chihuahua/
├── n02085782-Japanese_spaniel/
├── n02085936-Maltese_dog/
... (120+ breed folders)
```

## 📝 Scripts

| Script | Purpose |
|--------|---------|
| `train_model.py` | Train the XceptionNet model on dog breed dataset |
| `evaluate_model.py` | Evaluate model performance on test set |
| `live_detect.py` | Real-time breed detection from webcam |
| `resize_images.py` | Resize images to model input dimensions (299x299) |
| `normalize_images.py` | Normalize image pixel values |
| `split_data.py` | Split dataset into training and testing subsets |
| `data_generator.py` | Generate augmented data for training |
| `check_model.py` | Validate model architecture and weights |

## 📈 Results

The trained model provides:
- **Accuracy**: High classification accuracy on diverse dog breed images
- **Speed**: Real-time inference capability
- **Robustness**: Works with various image qualities and lighting conditions
- **Confidence Scores**: Provides probability scores for top predictions

## 🎯 Future Improvements

- [ ] Implement ensemble methods for higher accuracy
- [ ] Add more dog breeds to the classification
- [ ] Optimize model for mobile deployment
- [ ] Create a web interface
- [ ] Add attention visualization for interpretability
- [ ] Support for mixed breed detection

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Ujjwal Lutkarsh**
- GitHub: [@ujjwalutkarsh21](https://github.com/ujjwalutkarsh21)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This model is trained for dog breed classification. Accuracy may vary depending on image quality, lighting, and breed clarity. For production use, additional validation and testing is recommended.

---

**Note**: To use this project, ensure all image data is properly downloaded and the model weights file is present in the project root directory.
