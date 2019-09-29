import tensorflow as tf

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
from datetime import datetime
#from joblib import Parallel, delayed

parser = argparse.ArgumentParser('Train some data')

parser.add_argument('--bottleneck_dir',
                    help='base directory for bottlenecks')

args = parser.parse_args()

training = np.load(open('training/bottlenecks/batch-0001.npz', 'rb'))
train_data = training['MobileNetV2_bottleneck_features']
train_labels = training['yahoo_nsfw_output']

validation = np.load(open('training/bottlenecks/batch-0002.npz', 'rb'))
validation_data = validation['MobileNetV2_bottleneck_features']
validation_labels = validation['yahoo_nsfw_output']

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=train_data.shape[1:]))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=50,
          validation_data=(validation_data, validation_labels))
model.save_weights('bottleneck_fc_model.h5')