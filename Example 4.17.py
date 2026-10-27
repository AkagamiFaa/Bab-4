# Example 4.17 CNN Visualize Filters
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from matplotlib import pyplot as plt
from numpy import expand_dims

model = VGG19()
model.summary()

n=0
for layer in model.layers:
    print(n,layer.name)
    n+=1

n=0
for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(n,layer.name, filters.shape, biases.shape)
    n+=1

# visualize layer 1 filters
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters, ix = 4, 1
plt.figure(figsize=(10,10))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[0]):
        ax = plt.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[j,:, :], cmap='gray')
        ix += 1
plt.show()

def plot_feature_maps(feature_maps):
    col = 8
    row = int(feature_maps.shape[3]/col)
    ix = 1
    plt.figure(figsize=(20,20))
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    plt.show()

# feature maps of layer 1
model = VGG19()
n = 1
model = Model(inputs=model.inputs, outputs=model.layers[n].output)
try:
    img = load_img('Elephant.jpg', target_size=(224, 224))
    img = img_to_array(img)
    img = expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature_maps = model.predict(img)
    print("Feature maps: ", feature_maps.shape)
    plot_feature_maps(feature_maps)
except FileNotFoundError:
    print("Elephant.jpg not found for feature map generation.")
