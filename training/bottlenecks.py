#from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

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

#parser.add_argument('--base_dir',
#                    help='base directory for images')

parser.add_argument('--labels', 
                    help='text file with file names and associated labels')

parser.add_argument('--sample_size', type=int,
                    help='sample of images to use for testing purposes')

args = parser.parse_args()


save_dir = 'training/bottlenecks/{}'.format(datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
os.mkdir(save_dir)

def predict_and_save(x, y_true, batch_num):
    output = model.predict(x)
    full_file_name = os.path.join(save_dir, "batch-{}.npz".format(batch_num))
    np.savez_compressed(full_file_name, MobileNetV2_bottleneck_features=output, azure_output=y_true)

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
BATCH_SIZE = 100
#base_dir = args.base_dir
df=pandas.read_csv(args.labels)



datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

generator=datagen.flow_from_dataframe(
    directory='/',
    dataframe=df,
    x_col="file_name",
    y_col=['file_name', 'is_racy','racy_score','is_adult','adult_score'],
    class_mode='raw',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE)

model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False, 
    weights='imagenet')

batch_num = 1

for x, y_true in generator:
    print(x.shape)
    predict_and_save(x, y_true, batch_num)
    batch_num += 1
    print("Completed batch {}".format(batch_num))

    #Parallel(n_jobs=15)(delayed(runit)(image_file) for image_file in file_list)
