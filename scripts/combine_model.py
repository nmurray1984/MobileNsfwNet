from tensorflow.keras.models import load_model
import tensorflow as tf
import argparse
import tensorflowjs as tfjs


parser = argparse.ArgumentParser('Combines trained top model with full MobileNetV2 model')

parser.add_argument('--model',
                    help='model trained on top of MobileNetV2')

parser.add_argument('--target',
                    help='full model in TensorflowJS format')

args = parser.parse_args()

top_model = load_model(args.model, compile=False)

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False, 
    weights='imagenet')

model = tf.keras.Sequential([
  base_model,
  top_model
])

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


print(model.outputs)
print(model.inputs)

model.save(args.target)

#tfjs.converters.save_keras_model(model, args.target)
