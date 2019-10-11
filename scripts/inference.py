import tensorflow as tf

import os
import pandas
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse
import pickle
from datetime import datetime

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
#if args.sample_size is not None:
#df = df.head(100)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)

train_generator=datagen.flow_from_dataframe(
    directory=base_dir,
    dataframe=df,
    x_col="file",
    y_col=["nsfw_score"],
    class_mode="raw",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE)

val_generator=datagen.flow_from_dataframe(
    directory=base_dir,
    dataframe=df,
    x_col="file",
    y_col=["nsfw_score"],
    class_mode="raw",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation')

"""
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='validation')
"""
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
  tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(), #TODO check if this is correct
              loss='mean_absolute_error', 
              metrics=['accuracy'])

print(model.summary())

print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

epochs = 10

history = model.fit_generator(train_generator, 
                    epochs=epochs, 
                    validation_data=val_generator)

save_dir = 'training/models/{}'.format(datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
os.mkdir(save_dir)

model.save(os.path.join(save_dir, 'model.h5'))

with open(os.path.join(save_dir, 'history.pickled'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)



# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(save_dir, 'model_accuracy.png'))

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(save_dir, 'model_loss.png'))
