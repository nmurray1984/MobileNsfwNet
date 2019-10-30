
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import argparse
import glob
import os

parser = argparse.ArgumentParser('Inference on trained model')

parser.add_argument('--model',
                    help='model trained on top of MobileNetV2')

parser.add_argument('--base_dir',
                    help='folder to scan for images')

parser.add_argument('--save_npz', dest='save_npz', action='store_true')
parser.set_defaults(save_npz=False)

args = parser.parse_args()

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
BATCH_SIZE = 100

model = None
model_name = ""
if args.model is not None:
    model = load_model(args.model, compile=False)
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    model_name = os.path.basename(args.model).replace(".h5", "")
else:
    model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False, 
    weights='imagenet')
    model_name = "mobilenet_v2_1.0_224"

files = glob.glob(args.base_dir + '*/*.224x224.jpg')
print(len(files))
#files = files[0:10]

num_of_batches = int(len(files) / BATCH_SIZE) + 1 



for batch in range(0, num_of_batches):
    print('Starting batch {} of {}'.format(batch, num_of_batches))
    start = batch * BATCH_SIZE
    end = start + BATCH_SIZE - 1
    batch_files = files[start:end]
    images = np.empty([BATCH_SIZE, 224, 224, 3])
    i = 0
    for file_name in batch_files:
        img = image.load_img(file_name, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        x = image.img_to_array(img) / 255.
        images[i] = x
        i += 1
    classes = model.predict(images)
    
    for i in range(0, len(batch_files)):
        if(args.save_npz):
            original_file = batch_files[i]
            new_file = original_file.replace("224x224.jpg", model_name + "-bottleneck.npz")
            print("Saving file " + new_file)
            np.savez_compressed(new_file, bottleneck=classes[i])
        else:
            print('{},{},{},{}'.format(batch_files[i], classes[i][0], classes[i][1], classes[i][2]))
