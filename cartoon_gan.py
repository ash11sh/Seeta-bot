import cv2
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO

model_file = tf.keras.utils.get_file('cartoon_fp16.tflite','https://tfhub.dev/sayakpaul/lite-model/cartoongan/fp16/1?lite-format=tflite')

# Reference: https://www.tensorflow.org/lite/models/style_transfer/overview
def load_img(path_to_img):
  img = cv2.imread(path_to_img)
  img = img.astype(np.float32) / 127.5 - 1
  img = np.expand_dims(img, 0)
  img = tf.convert_to_tensor(img)
  return img

# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim=224):
  # Resize the image so that the shorter dimension becomes the target dim.
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  image = tf.image.resize(image, (target_dim,target_dim))
  # Central crop the image.
  # image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
  return image




def cartoon_model(source_image):
  model_type = "float16" 
  source_image = load_img(source_image)
  base_shape = tf.shape(source_image)
  preprocessed_source_image = preprocess_image(source_image, target_dim=224) 

  interpreter = tf.lite.Interpreter(model_path=model_file)
  input_details = interpreter.get_input_details()
  interpreter.allocate_tensors()
  interpreter.set_tensor(input_details[0]['index'], preprocessed_source_image)
  interpreter.invoke()


  raw_prediction = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
  output = (np.squeeze(raw_prediction)+1.0)*127.5
  output = np.clip(output, 0, 255).astype(np.uint8)
  output = cv2.resize(output, (base_shape[2],base_shape[1]))
  image=Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
  bio = BytesIO()
  bio.name = 'cartoongan.jpg'
  image.save(bio, 'JPEG')
  return bio