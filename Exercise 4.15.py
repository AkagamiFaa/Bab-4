#Exercise 4.15
# Select the second convolutional layer to visualize by changing n = 1 to n = 2.
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot as plt

model = VGG19()

# Use n = 2 for second convolutional layer
n = 2
filters, biases = model.layers[n].get_weights()
s = filters.shape
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters, ix = 4, 1
plt.figure(figsize=(10,10))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[0]): # 64 channels in layer 2, plotting just a few or it will be too large
        # We will limit j to first 3 channels for visualization purposes
        if j < 3:
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[j,:, :], cmap='gray')
            ix += 1
plt.show()
