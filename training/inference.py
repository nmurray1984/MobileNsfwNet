import tensorflow as tf

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser('Train some data')

parser.add_argument('--base_dir',
                    help='base directory for images')

parser.add_argument('--labels', 
                    help='text file with file names and associated labels')

parser.add_argument('--sample_size', type=int,
                    help='sample of images to use for testing purposesß')

args = parser.parse_args()


base_dir = args.base_dir

IMAGE_SIZE = 224
BATCH_SIZE = 64

df=pandas.read_csv(args.labels, names=('file','sfw_score','nsfw_score'))

#to reduce testing size
if args.sample_size > 0:
    df = df.head(args.sample_size)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)

train_generator=datagen.flow_from_dataframe(
    directory=base_dir,
    dataframe=df,
    x_col="file",
    y_col=["sfw_score", "nsfw_score"],
    class_mode="raw",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE)

val_generator=datagen.flow_from_dataframe(
    directory=base_dir,
    dataframe=df,
    x_col="file",
    y_col=["sfw_score", "nsfw_score"],
    class_mode="raw",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation')

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False, 
    weights='imagenet')

base_model.trainable = False

# if predicting, this would be used
#predictions = base_model.predict(
#    x=val_generator
#)

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(2, activation='relu')
])

model.compile(optimizer=tf.keras.optimizers.Adam(), #TODO check if this is correct
              loss='mean_squared_error', 
              metrics=['accuracy'])

predictions = model.predict(
    x=val_generator
)

print(predictions)

print(model.summary())

print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

epochs = 10

history = model.fit_generator(train_generator, 
                    epochs=epochs, 
                    validation_data=val_generator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()