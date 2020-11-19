import os
import cv2
import time
import argparse
import numpy as np
from PIL import Image
from skimage.filters import gaussian
from io import BytesIO


try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter


LABEL_CONTOURS = [(0, 0, 0),  # 0=background
                  # 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye
                  (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  # 6=eye_g, 7=l_ear, 8=r_ear, 9=ear_r, 10=nose
                  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                  # 11=mouth, 12=u_lip, 13=l_lip, 14=neck, 15=neck_l
                  (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                  # 16=cloth, 17=hair, 18=hat
                  (0, 64, 0), (128, 64, 0), (0, 192, 0)]

def decode_prediction_mask(mask):
    mask_shape = mask.shape
    mask_color = np.zeros(shape=[mask_shape[0], mask_shape[1], 3], dtype=np.uint8)
    unique_label_ids = [v for v in np.unique(mask) if v != 0 and v != 255]
    for label_id in unique_label_ids:
        idx = np.where(mask == label_id)
        mask_color[idx] = LABEL_CONTOURS[label_id]
    return mask_color


def sharpen(img):
	img = img * 1.0
	gauss_out = gaussian(img, sigma=5, multichannel=True)

	alpha = 1.5
	img_out = (img - gauss_out) * alpha + img

	img_out = img_out / 255.0

	mask_1 = img_out < 0
	mask_2 = img_out > 1

	img_out = img_out * (1 - mask_1)
	img_out = img_out * (1 - mask_2) + mask_2
	img_out = np.clip(img_out, 0, 1)
	img_out = img_out * 255
	return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[ 255,0,0]):
	b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
	tar_color = np.zeros_like(image)
	tar_color[:, :, 0] = b
	tar_color[:, :, 1] = g
	tar_color[:, :, 2] = r

	image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

	if part == 12 or part == 13:
		image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
	else:
		# image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]
		image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]  
	changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
	changed[parsing != part] = image[parsing != part]

	return changed




def hair_pred(path,color):
  img = cv2.imread(path) 
  deep_model='bisenetv2_celebamaskhq_448x448_float16_quant.tflite'
  num_threads=4
  interpreter = Interpreter(model_path=deep_model, num_threads=num_threads)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()[0]['index']
  bisenetv2_predictions = interpreter.get_output_details()[0]['index']
  model_height = interpreter.get_input_details()[0]['shape'][1]
  model_width = interpreter.get_input_details()[0]['shape'][2]
  height, width, channels = img.shape


  prepimg_deep_size = cv2.resize(img, (model_width, model_height))
  prepimg_deep = cv2.cvtColor(prepimg_deep_size, cv2.COLOR_BGR2RGB)
  prepimg_deep = np.expand_dims(prepimg_deep, axis=0)
  prepimg_deep = prepimg_deep.astype(np.float32)  
  prepimg_deep /= 255.0
  prepimg_deep -= [[[0.5, 0.5, 0.5]]]
  prepimg_deep /= [[[0.5, 0.5, 0.5]]]


  # Run model

  interpreter.set_tensor(input_details, prepimg_deep)
  interpreter.invoke()

  # Get results
  predictions = interpreter.get_tensor(bisenetv2_predictions)
  table = {
		'hair': 17,
		'upper_lip': 12,
		'lower_lip': 13
	}

  part = table['hair']
  image = hair(prepimg_deep_size , predictions, part, color)
  imdraw = cv2.resize(image, (width,height))
  image=Image.fromarray(cv2.cvtColor(imdraw, cv2.COLOR_BGR2RGB))
  bio = BytesIO()
  bio.name = 'hair_seg.jpg'
  image.save(bio, 'JPEG')
  return bio
