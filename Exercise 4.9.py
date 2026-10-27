#Exercise 4.9
# Load VGG19 instead of VGG16
from tensorflow.keras.applications.vgg19 import VGG19
model = VGG19(weights='imagenet')
print(model.summary())
