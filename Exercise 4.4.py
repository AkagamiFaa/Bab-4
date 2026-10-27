#Exercise 4.4
# Modify Example 4.4 to use a different number of filters to see effect on parameters.
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D

model = Sequential()
# Changed filters from 6 to 12
model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
model.add(AveragePooling2D(pool_size=(2, 2)))
# Changed filters from 16 to 32
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# The number of parameters will increase due to more filters
model.summary()
