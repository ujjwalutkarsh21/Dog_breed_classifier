from tensorflow.keras.models import load_model

# Load saved model
model = load_model("breed_classifier_xception.h5")


model.summary()
