from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import backend as K
import os
import numpy as np
import matplotlib.pyplot as pl
import argparse

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

parser = argparse.ArgumentParser('Combines trained top model with full MobileNetV2 model')

parser.add_argument('--image_dir',
                    help='directory for images')

parser.add_argument('--target',
                    help='where to save the final model in h5 format')

args = parser.parse_args()

IMAGE_SIZE = 224
BATCH_SIZE = 64

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)

train_generator = datagen.flow_from_directory(
    args.image_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')

val_generator = datagen.flow_from_directory(
    args.image_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='validation')

for image_batch, label_batch in train_generator:
  break

print (train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:
  f.write(labels)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(2, activation='softmax')
])

def prediction_min(y_true, y_pred):
    final = K.min(y_pred)
    return final

def prediction_max(y_true, y_pred):
    final = K.max(y_pred)
    return final

def prediction_variance(y_true, y_pred):
    final = K.var(y_pred)
    return final

model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

epochs = 2

history = model.fit_generator(train_generator, 
                    epochs=epochs, 
                    validation_data=val_generator)

model.save(args.target)