import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np

# Load model
model = load_model("breed_classifier_xception.h5")

# Recreate class label mapping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.4)
train_gen = datagen.flow_from_directory(
    "resized_images",
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


