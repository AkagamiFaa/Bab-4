#Exercise 4.10
# Use VGG19 for classification
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

model = VGG19(weights='imagenet')
img_path = 'Elephant.jpg'
try:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    results = decode_predictions(predictions)
    print(results)
except FileNotFoundError:
    print(f"Image {img_path} not found. Please place an image to test.")
