# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.xception import preprocess_input
# import numpy as np

# # Load the trained model
# model = load_model("breed_classifier_xception.h5")

# # (Optional) Recreate your ImageDataGenerator and train_gen for class labels if needed
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     validation_split=0.4
# )

# train_gen = datagen.flow_from_directory(
#     "resized_images",
#     target_size=(299, 299),
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'
# )

# # Evaluate on validation set (requires validation_gen setup if you want to use it)
# validation_gen = datagen.flow_from_directory(
#     "resized_images",
#     target_size=(299, 299),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'
# )

# val_loss, val_acc = model.evaluate(validation_gen)
# print("Validation Accuracy:", val_acc)
# print("Validation Loss:", val_loss)

# # Predict random dog image
# img = load_img("my_dog.jpg", target_size=(299, 299))  # Update filename if needed
# img_array = img_to_array(img)
# img_array = preprocess_input(img_array)
# img_array = np.expand_dims(img_array, axis=0)

# pred = model.predict(img_array)
# breed_idx = np.argmax(pred)
# class_labels = list(train_gen.class_indices.keys())
# print("Predicted breed:", class_labels[breed_idx])



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np

model = load_model("breed_classifier_xception.h5")

# Recreate train_gen ONLY for class labels mapping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.4)
train_gen = datagen.flow_from_directory(
    "resized_images",
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Predict random dog image (in root folder)
img = load_img("my_dog.jpg", target_size=(299, 299))
img_array = img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
breed_idx = np.argmax(pred)
class_labels = list(train_gen.class_indices.keys())
print("Predicted breed:", class_labels[breed_idx])
