#Example 4.11
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # sembunyikan log info/warning TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # sembunyikan pesan oneDNN
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
model = VGG16(weights='imagenet')
img_path = 'Elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# prediction
predictions = model.predict(x, verbose=0)
results = decode_predictions(predictions)
for kode, label, prob in results[0]:
    print(f"{label:20s} : {prob*100:.2f}%")
