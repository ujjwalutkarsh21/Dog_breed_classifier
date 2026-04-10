import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
import os

# Define project root (BASE_DIR)
# This script is in scripts/BASE_DIR
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "breed_classifier_xception.h5")
DATA_DIR = os.path.join(BASE_DIR, "resized_images")

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Recreate class label mapping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.4)

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found at: {DATA_DIR}")

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
class_labels = list(train_gen.class_indices.keys())

# Open webcam
cap = cv2.VideoCapture(0)  # 0 is default laptop/USB webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for prediction
    img = cv2.resize(frame, (299, 299))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_to_array(img_rgb)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    breed_idx = np.argmax(pred)
    breed = class_labels[breed_idx]

    # Annotate and display frame
    cv2.putText(frame, f"Breed: {breed}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Dog Breed Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


