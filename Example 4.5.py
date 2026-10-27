#Example 4.5 AlexNet with Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), activation='relu', strides=(4,4), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(filters=256, kernel_size=(11,11), activation='relu', strides=(1,1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))
model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(17, activation='softmax'))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
