# Data generator setup
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.4)

train_gen = datagen.flow_from_directory(
    "resized_images",
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_gen = datagen.flow_from_directory(
    "resized_images",
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model definition (your supplied code)
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_gen,
    epochs=10,
    validation_data=validation_gen
)


model.save("breed_classifier_xception.h5")
print("Model saved as breed_classifier_xception.h5")
