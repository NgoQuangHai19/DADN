import numpy as np
import os 
from PIL import Image
import tensorflow as tf 
from tensorflow.keras import layers,models
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


data_path = 'train_data'
numpy_images = []
labels = []
original_labels = os.listdir(data_path)

folder_images = [os.path.join(data_path, f) for f in os.listdir(data_path)]
for idx,folder in enumerate(folder_images):
  label = idx

  images = [os.path.join(folder, f) for f in os.listdir(folder)]
  for image in images:
    face_image = Image.open(image)

    # resize the image
    face_image = face_image.resize((128, 128))
    face_numpy = np.array(face_image, 'uint8')

    numpy_images.append(face_numpy)
    labels.append(label)

numpy_images = np.array(numpy_images)
numpy_images = preprocess_input(numpy_images) 
labels = np.array([labels]).T
labels = to_categorical(labels, num_classes=len(original_labels))
# split data
X_train, X_val, y_train, y_val = train_test_split(numpy_images, labels, test_size=0.15, random_state=42)
# X_train = preprocess_input(X_train) 
# y_train = preprocess_input(y_train)
# X_val = to_categorical(X_val, num_classes=len(original_labels))
# y_val = to_categorical(y_val, num_classes=len(original_labels))

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)


based_model = vgg16.VGG16(weights = 'imagenet',
                    include_top = False,
                    input_shape = (128, 128, 3))

# Freeze layers, not training these layers
for layer in based_model.layers:
    layer.trainable = False 

model = models.Sequential()

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(128, activation='relu')
dense_layer_2 = layers.Dense(64, activation='relu')
prediction_layer = layers.Dense(len(original_labels), activation='softmax')

model = models.Sequential([
    based_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)


es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)

model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), batch_size=32, callbacks=[es])

model.save('face_id.h5')



