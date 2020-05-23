import os
import cv2
import copy
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D
from keras.models import load_model

def load_data(path, input_size, f1, f3):
    x = []
    y = []
    center_bias = int((f1 + f3 - 2) / 2)
    half_size = int((input_size - 1) / 2 + 1)
    begin = center_bias
    end = input_size - center_bias
    files = os.listdir(path)
    image_postfix = ['jpg', 'png']
    images = [file for file in files if file.split('.')[-1] in image_postfix]
    for file in images:
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

def train():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    path = 'images'
    c = 1
    f1, f2, f3 = 9, 1, 5
    # n1, n2 = 64, 32
    n1, n2 = 128, 64
    input_size = 33
    target_size = input_size - f1 - f3 + 2
    batch_size = 256
    epochs = 64
    x, y = load_data(path, input_size, f1, f3)
    if K.image_data_format() == 'channels_first':
        x = x.reshape(x.shape[0], 1, input_size, input_size)
        y = y.reshape(y.shape[0], 1, target_size, target_size)
        input_shape = (1, None, None)
    else:
        x = x.reshape(x.shape[0], input_size, input_size, 1)
        y = y.reshape(y.shape[0], target_size, target_size, 1)
        input_shape = (None, None, 1)

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

    # model.save('srcnn.h5')
    model.save_weights('srcnn.h5')

def get_psnr_of_Y(origin_image, super_res):
    o_y = origin_image[:, :, 0]
    sr_y = super_res[:, :, 0]
    diff = o_y - sr_y
    mse = np.mean(np.square(diff))
    psnr = 10 * np.log10(255 * 255 / mse)
    return round(psnr, 3)

def get_predict_model():
    # K.reset_default_graph()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    c = 1
    f1, f2, f3 = 9, 1, 5
    n1, n2 = 64, 32
    if K.image_data_format() == 'channels_first':
        input_shape = (1, None, None)
    else:
        input_shape = (None, None, 1)
    model = Sequential()
    model.add(Conv2D(n1, (f1, f1), activation='relu', input_shape=input_shape, padding='same'))
    model.add(Conv2D(n2, (f2, f2), activation='relu'))
    model.add(Conv2D(c, (f3, f3), padding='same'))
    model.load_weights('srcnn.h5')
    return model

def super_resolution(image):
    # image_path = 'images/wallhaven-odd2om.jpg'
    # image = np.array(Image.open(image_path).convert("YCbCr"))
    height, width, _ = image.shape
    r_Y = cv2.resize(image[:, :, 0], (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    r_Y = r_Y.astype('float32') / 255
    # r_Cb = cv2.resize(image[:, :, 1], (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    # r_Cr = cv2.resize(image[:, :, 2], (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    r_image = cv2.resize(image[:, :, :], (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    if K.image_data_format() == 'channels_first':
        r_Y = r_Y.reshape(1, 1, height * 2, width * 2)
    else:
        r_Y = r_Y.reshape(1, height * 2, width * 2, 1)
    # model = load_model('srcnn.h5', custom_objects={'my_loss': my_loss})
    model = get_predict_model()
    pred_Y = model.predict(r_Y)
    pred_Y = pred_Y.reshape(height * 2, width * 2)
    pred_Y[pred_Y > 1] = 1
    pred_Y[pred_Y < 0] = 0
    pred_Y = pred_Y * 255
    r_image[:, :, 0] = pred_Y
    # img_bgr = cv2.cvtColor(r_image, cv2.COLOR_YCrCb2RGB)

    # file_name = ''.join(image_path.split('/')[-1].split('.')[:-1])
    # result_path = 'results/{}-rs.jpg'.format(file_name)
    # cv2.imwrite(result_path, img_bgr)
    return r_image


def test_image_score(image, file):
    height, width, _ = image.shape
    low_res_image = cv2.resize(image[:, :, :], (width>>1, height>>1), interpolation=cv2.INTER_CUBIC)
    super_res_image = super_resolution(low_res_image)
    nearest_res_image = cv2.resize(low_res_image[:, :, :], (width, height), interpolation=cv2.INTER_NEAREST)
    linear_res_image = cv2.resize(low_res_image[:, :, :], (width, height), interpolation=cv2.INTER_LINEAR)
    area_res_image = cv2.resize(low_res_image[:, :, :], (width, height), interpolation=cv2.INTER_AREA)
    cubic_res_image = cv2.resize(low_res_image[:, :, :], (width, height), interpolation=cv2.INTER_CUBIC)
    lanczos4_res_image = cv2.resize(low_res_image[:, :, :], (width, height), interpolation=cv2.INTER_LANCZOS4)
    srcnn_psnr = get_psnr_of_Y(image, super_res_image)
    nearest_psnr = get_psnr_of_Y(image, nearest_res_image)
    linear_psnr = get_psnr_of_Y(image, linear_res_image)
    area_psnr = get_psnr_of_Y(image, area_res_image)
    cubic_psnr = get_psnr_of_Y(image, cubic_res_image)
    lanczos4_psnr = get_psnr_of_Y(image, lanczos4_res_image)
    print('{} srcnn: {}dB, nearest: {}dB, linear: {}dB, cubic: {}dB'.format(file, srcnn_psnr, nearest_psnr, linear_psnr, cubic_psnr))
    with open('scores.csv', 'a') as f:
        f.write('{},{},{},{},{}\n'.format(file, srcnn_psnr, nearest_psnr, linear_psnr, cubic_psnr))

import os
from multiprocessing import Process
def test_model():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    path = 'images'
    files = os.listdir(path)
    image_postfix = ['jpg', 'png']
    images = [file for file in files if file.split('.')[-1] in image_postfix]
    with open('scores.csv', 'w') as f:
        f.write('file,srcnn,nearest,linear,cubic\n')

    for file in images:
        print(file)
        image_path = os.path.join(path, file)
        image = np.array(Image.open(image_path).convert("YCbCr"))
        height, width, _ = image.shape
        if height % 2 or width % 2:
            height = height>>1<<1
            width = width>>1<<1
            image = cv2.resize(image[:, :, :], (width, height), interpolation=cv2.INTER_CUBIC)
        p = Process(target = test_image_score, args = (image, file, ))
        p.start()
        p.join()

if __name__ == '__main__':
    train()
    # predict()
    # model = get_predict_model()
    # test_model()