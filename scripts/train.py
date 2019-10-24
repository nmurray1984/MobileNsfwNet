import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas
import numpy as np # linear algebra
import os # accessing directory structure
import os
import glob

import argparse

parser = argparse.ArgumentParser('Train from a csv')

parser.add_argument('--save_model_file',
                    help='save_model_file')

parser.add_argument('--csv_file',
                    help='file with list of files and their classes')

args = parser.parse_args()

#ensures repeatability of experiments
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

df=pandas.read_csv(args.csv_file)

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
        dataframe=df[0:39999],
        directory='',
        x_col="file_name",
        y_col="new_class",
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

validation_generator = datagen.flow_from_dataframe(
        dataframe=df[40000:47750],
        directory='',
        x_col="file_name",
        y_col="new_class",
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                              include_top=False, 
                                              weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(7, 7, 1280)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(3,activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit_generator(train_generator, 
                    epochs=10, 
                    validation_data=validation_generator)

model.save('output.h5')

