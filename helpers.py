import flickrapi
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.applications import VGG16
from keras.preprocessing import image as img
from keras_applications.vgg16 import preprocess_input
from tensorflow.python.keras.applications.vgg16 import decode_predictions
import os

model = VGG16(weights='imagenet', include_top=True)
model.summary()

# Load image and center-crop it
image = Image.open('data/pug.jpeg')
width, height = image.size
square_size = min(width, height)
x = (width - square_size) // 2
y = (height - square_size) // 2

image = image.crop((x, y, square_size, square_size))
plt.imshow(np.asarray(image))

# Resizes to the networks required input size.
target_square_size = max(dimension for dimension in model.layers[0].input_shape if dimension)
image = image.resize((target_square_size, target_square_size), Image.ANTIALIAS)
plt.imshow(np.asarray(image))

numpy_image = img.img_to_array(image)
print(numpy_image.shape)
image_batch = np.expand_dims(numpy_image, axis=0)
print(image_batch.shape)
pre_processed = preprocess_input(image_batch, data_format='channels_last')
print(pre_processed.shape)


features = model.predict(pre_processed)
print(features.shape)

import pprint

pprint.pprint(decode_predictions(features, top=10))

FLCIKR_KEY = os.environ['FLICKR_KEY']
FLICKR_SECRET = os.environ['FLICKR_SECRET']

flickr = flickrapi.FlickrAPI(FLCIKR_KEY, FLICKR_SECRET, format='parsed-json')
results = flickr.photos.search(text='cat', per_page='10', sort='relevance')
photos = results['photos']['photo']
pprint.pprint(results)
