import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
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

class NumPyFileGenerator(Sequence):
    def __init__(self, file_list, batch_size):
        self.file_list = file_list
        self.class_list = file_list['new_class'].tolist()
        self.file_name_list = file_list['file_name'].tolist()
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.file_name_list) / self.batch_size)

    def get_bottleneck_file_name(self, original):
        dirname = os.path.dirname(original)
        basename = os.path.basename(original)
        #hardcoding for now
        model = 'mobilenet_v2_1.0_224'
        new_basename = basename.replace('224x224.jpg', model + '-bottleneck.npz')
        return os.path.join(dirname, new_basename)

    def __getitem__(self, idx):
        x = np.empty((0, 7, 7, 1280))
        y = np.empty((0))
        for i in range(idx * self.batch_size, ((idx + 1) * self.batch_size)):
                file_name = self.file_name_list[i]
                npz_file = self.get_bottleneck_file_name(file_name)
                bottleneck_data = np.load(npz_file, allow_pickle=True)
                x_row = bottleneck_data['bottleneck']
                x = np.append(x, [x_row], axis=0)
                y_row = 0
                if self.class_list[i] == 'sunflowers':
                        y_row = 1
                y = np.append(y, [y_row], axis=0)
        return x, to_categorical(y, num_classes=2)

VALIDATION_PERCENT = .20

train_start = 0
train_end = int(len(df) * (1 - VALIDATION_PERCENT))
validation_start = train_end + 1
validation_end = len(df)
print(train_end)
print(validation_end)
train_generator = NumPyFileGenerator(df[train_start:train_end], 32)
validation_generator = NumPyFileGenerator(df[validation_start:validation_end], 32)

#base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
#                                              include_top=False, 
#                                              weights='imagenet')

#base_model.trainable = False

model = tf.keras.Sequential([
  #base_model,
  tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(7, 7, 1280)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(2,activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit_generator(train_generator, 
                    epochs=10, 
                    validation_data=validation_generator)

model.save('output.h5')

