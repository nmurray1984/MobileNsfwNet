from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils import Sequence
from keras import backend as K
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
import glob


#ensures repeatability of experiments
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

class NumPyFileGenerator(Sequence):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(open(self.files[idx], 'rb'),allow_pickle=True)
        x = data['MobileNetV2_bottleneck_features']
        a = data['azure_output']
        b = map(lambda x: 1 if x else 0, a[:,3].tolist())
        y = list(b)
        return x, y

all_files = glob.glob("/kaggle/input/*/*/*/*/*.npz")
len(all_files)

training_generator = NumPyFileGenerator(files=all_files[0:2999])
validation_generator = NumPyFileGenerator(files=all_files[3000:3475])

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(7, 7, 1280)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', 
                       tf.keras.metrics.FalsePositives(), 
                       tf.keras.metrics.FalseNegatives(),
                       binary_accuracy_50, 
                       binary_accuracy_60, 
                       binary_accuracy_70, 
                       binary_accuracy_80, 
                       binary_accuracy_90])

history = model.fit_generator(training_generator, 
                    epochs=10, 
                    validation_data=validation_generator)

model.save('output.h5')

