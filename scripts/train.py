import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import pandas
import numpy as np # linear algebra
import os # accessing directory structure
import os
import glob
from sklearn.metrics import classification_report, confusion_matrix
import argparse

parser = argparse.ArgumentParser('Train from a csv')

parser.add_argument('--csv_file',
                    help='file with list of files and their classes')

parser.add_argument('--inference_only',dest='inference_only', action='store_true')
parser.set_defaults(inference_only=False)

parser.add_argument('--model', help="Model file to load instead of clean start")

parser.add_argument('--use_bottlenecks',dest='use_bottlenecks', action='store_true')
parser.set_defaults(use_bottlenecks=False)

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
                if self.class_list[i] == '1-racy':
                        y_row = 1
                elif self.class_list[i] == '2-adult':
                        y_row = 2
                y = np.append(y, [y_row], axis=0)
        return x, to_categorical(y, num_classes=3)

class ImageFileGenerator(Sequence):
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
        x = np.empty((0, 224, 224, 3))
        y = np.empty((0))
        for i in range(idx * self.batch_size, ((idx + 1) * self.batch_size)):
                file_name = self.file_name_list[i]
                img = image.load_img(file_name, target_size=(224, 224))
                x_row = image.img_to_array(img) / 255.
                x = np.append(x, [x_row], axis=0)
                y_row = 0
                if self.class_list[i] == '1-racy':
                        y_row = 1
                elif self.class_list[i] == '2-adult':
                        y_row = 2
                y = np.append(y, [y_row], axis=0)
        return x, to_categorical(y, num_classes=3)

VALIDATION_PERCENT = .30

train_generator = None
validation_generator = None

train_start = 0
train_end = int(len(df) * (1 - VALIDATION_PERCENT))
validation_start = train_end + 1
validation_end = len(df)

if args.use_bottlenecks:
    train_generator = NumPyFileGenerator(df[train_start:train_end], 32)
    validation_generator = NumPyFileGenerator(df[validation_start:validation_end], 32)
else:
    train_generator = ImageFileGenerator(df[train_start:train_end], 32)
    validation_generator = ImageFileGenerator(df[validation_start:validation_end], 32)

#base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
#                                              include_top=False, 
#                                              weights='imagenet')

#base_model.trainable = False

model = None

if(args.model):
    model = load_model(args.model, compile=False)
    if(args.use_bottlenecks):
        model = model[1] #use the top layer
else:
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
    if(args.use_bottlenecks):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(7, 7, 1280)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3,activation='softmax')
        ])  

def test_custom_metric(y_true, y_pred):
    act = K.argmax(y_true, axis=0)
    pred = K.argmax(y_pred, axis=0)
    #adult_correct = K.sum(K.all(K.concatenate([act >= 2, pred >= 2], axis=1), dtype='float16'))

    adult_correct = K.sum(K.cast(K.equal(act >= 2,pred >= 2), dtype='float16'))
    adult_act = K.sum(K.cast(act >= 2, dtype='float16'))
    final = adult_correct / adult_act
    #final = K.mean(K.cast(pred >= 2, dtype='float16') / K.cast(act >= 2, dtype='float16')
    return final

model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy', test_custom_metric])

if not args.inference_only:
    history = model.fit_generator(train_generator, 
                    epochs=1, 
                    validation_data=validation_generator)

    model.save('output.h5')

def mapitems(item):
   if item == '1-racy':
      return 1
   elif item == '2-adult':
      return 2
   else:
      return 0

print('Building confusion matrix')
Y_pred = model.predict_generator(validation_generator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
np.save('predictions', Y_pred)
Y_true = df[validation_start:validation_end]['new_class'].tolist()
y_true = list(map(mapitems, Y_true))
np.save('actuals', y_true) 
total_predictions = Y_pred.shape[0]
print(confusion_matrix(y_true[0:total_predictions], y_pred[0:total_predictions]))
