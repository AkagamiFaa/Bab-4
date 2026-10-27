#Example 4.12
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from tensorflow.keras.applications import mobilenet

model = mobilenet.MobileNet(weights='imagenet')
print(model.summary())

img_path = 'Elephant.jpg'
try:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    processed_image = mobilenet.preprocess_input(x)
    predictions = model.predict(processed_image)
    results = decode_predictions(predictions)
    print(results)
except FileNotFoundError:
    print(f"Image {img_path} not found. Please place an image to test.")
