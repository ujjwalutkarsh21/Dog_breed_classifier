""" Normalization was used with the help of Data generator setup (Code commeted out below as it was
    already existed in train_model.py script)"""


# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.xception import preprocess_input

# datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.4)

# train_gen = datagen.flow_from_directory(
#     "resized_images",
#     target_size=(299, 299),
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'
# )
# validation_gen = datagen.flow_from_directory(
#     "resized_images",
#     target_size=(299, 299),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'
# )