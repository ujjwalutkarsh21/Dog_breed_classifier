from tensorflow.keras.models import load_model

# Load your saved model
model = load_model("breed_classifier_xception.h5")

# Print model summary
model.summary()
