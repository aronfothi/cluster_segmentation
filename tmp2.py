'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
from keras.callbacks import TensorBoard

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


print('y_train shape:', y_train.shape)

def build_sub_model(input_shape):

    #model = Sequential()
    inpa = Input(input_shape, name="sub_inpa")
    inpb = Input(input_shape, name="sub_inpb")
    
    x = Conv2D(32, kernel_size=(3, 3),
                    activation='relu')(inpa)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model([inpa, inpb], [x, inpb], name="sub_model")
    print(model.summary())
    print(model.inputs)
    return model

def build_model():


    sub_model = build_sub_model(input_shape)

    inp2a = Input(input_shape, name="inp2a")
    inp2b = Input(input_shape, name="inp2b")

    inputs = [inp2a, inp2b]
    sub_out = sub_model(inputs)
    #print(len(x_inpb), x_inpb)
    model = Model(inputs, sub_out)
    print(model.summary())
    print(model.get_layer("sub_model").inputs)
    model.get_layer("sub_model").inputs = model.inputs
    print(model.summary())
    print(model.get_layer("sub_model").summary())
    print(model.get_layer("sub_model").inputs)
    for i in model.get_layer("sub_model").layers:
        if isinstance(i, InputLayer):
            print('sub', i, i.name, i.get_output_at(0))
    print(model.inputs)
    for i in model.layers:
        if isinstance(i, InputLayer):
            print('main', i, i.name, i.get_output_at(0))
    
    return model

model2 =  build_model()

model2.compile(loss=[keras.losses.categorical_crossentropy, keras.losses.mean_squared_error],
               loss_weights=[1., 1.],
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
callbacks = [
             TensorBoard(log_dir='/home/fothar/cluster_segmentation_redesign/logs/xxx/',
                                        histogram_freq=0, write_graph=True, write_images=False)
        ]              

model2.fit([x_train, x_train], [y_train, x_train],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=([x_test, x_test], [y_test, x_test]))
score = model2.evaluate([x_test, x_test], [y_test, x_test], verbose=0)
print(len(score), score)
