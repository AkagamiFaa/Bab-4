#Exercise 4.16
# Display all filters by changing n_filters, ix = 4, 1 to n_filters, ix = s[3], 1.
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot as plt

model = VGG19()

n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters, ix = s[3], 1
plt.figure(figsize=(20,20))
for i in range(n_filters): # s[3] = 64
    f = filters[:, :, :, i]
    for j in range(s[0]): # s[0] = 3
        ax = plt.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[j,:, :], cmap='gray')
        ix += 1
plt.show()
