import os
import cv2
import copy
import keras
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D
from keras import backend as K

def load_data(path, input_size, f1, f3):
    x = []
    y = []
    center_bias = int((f1 + f3 - 2) / 2)
    half_size = int((input_size - 1) / 2 + 1)
    begin = center_bias
    end = input_size - center_bias
    files = os.listdir(path)
    for file in files:
        image_path = os.path.join(path, file)
        image = np.array(Image.open(image_path).convert("YCbCr"))
        height, width, channel = image.shape
        print(height, width, channel)
        for i in range(0, height, input_size):
            if i + input_size - 1 >= height:
                continue
            for j in range(0, width, input_size):
                if j + input_size - 1 >= width:
                    continue
                patch = copy.deepcopy(image[i:i+input_size, j:j+input_size])
                center = copy.deepcopy(patch[begin:end, begin:end])[:, :, 0]
                patch = cv2.resize(patch, (half_size, half_size), interpolation=cv2.INTER_CUBIC)
                patch = cv2.resize(patch, (input_size, input_size), interpolation=cv2.INTER_CUBIC)[:, :, 0]
                x.append(patch)
                y.append(center)
    x = np.array(x).astype('float32') / 255
    y = np.array(y).astype('float32') / 255
    return x, y

def my_loss(y_true, y_pred):
    y_true = K.reshape(y_true, [1, -1])
    y_pred = K.reshape(y_pred, [1, -1])
    return K.mean(K.square(y_pred - y_true))

if __name__ == '__main__':
    path = 'images'
    c = 1
    f1, f2, f3 = 9, 1, 5
    n1, n2 = 64, 32
    input_size = 33
    target_size = input_size - f1 - f3 + 2
    batch_size = 32
    epochs = 12
    x, y = load_data(path, input_size, f1, f3)
    if K.image_data_format() == 'channels_first':
        x = x.reshape(x.shape[0], 1, input_size, input_size)
        y = y.reshape(y.shape[0], 1, target_size, target_size)
        input_shape = (1, input_size, input_size)
    else:
        x = x.reshape(x.shape[0], input_size, input_size, 1)
        y = y.reshape(y.shape[0], target_size, target_size, 1)
        input_shape = (input_size, input_size, 1)

    model = Sequential()
    # Level 1 64 Conv Filters of 9 * 9
    model.add(Conv2D(n1, (f1, f1), activation='relu', input_shape=input_shape))
    # Level 2 32 Conv Filters of 1 * 1
    model.add(Conv2D(n2, (f2, f2), activation='relu'))
    # Level 3 1 Conv Filter of 5 * 5
    model.add(Conv2D(c, (f3, f3)))

    model.compile(loss=my_loss,
              optimizer=keras.optimizers.SGD(),
              metrics=['mean_squared_error'])

    model.fit(x, y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          verbose=1)