"""
This script is for model compiling, training, and saving
"""
from keras.callbacks import EarlyStopping
from keras.layers import Cropping2D
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
import pandas as pd
import numpy as np

'''
VGG - 16 from scratch
'''
def vgg16(input_shape):
    model = Sequential()

    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


"""
Load VGG-16 model and its weight as base model from tensorflow keras application without the full connection layer
weights from ImageNet, and freeze the model
append the previous full connection layer to the base model
"""
def vgg_pretrained(input_shape, crop, fine_tuning=False):
    size = input_shape[0] - (2 * crop)
    pretrained_model = PretrainedModel(
        input_shape=(size, size, 3),
        weights='imagenet',
        include_top=False
    )

    if fine_tuning:
        for layer in pretrained_model.layers:
            if layer.name in ['block5_conv1', 'block4_conv1']:
                layer.trainable = True
            else:
                layer.trainable = False
    else:
        pretrained_model.trainable = False

    model = Sequential()
    model.add(Cropping2D(cropping=((crop, crop), (crop, crop)), input_shape=input_shape))
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

"""
Train the given model with given data and parameter
Save the trained model in saved_model
Save the history object as csv in History
"""
def train_save(model, name, train, val, epochs, batch, train_step, val_step):
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30, verbose=1, mode='auto')
    history = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
        batch_size=batch,
        steps_per_epoch=train_step,
        validation_steps=val_step,
        callbacks=[early]
    )
    model.save('./saved_model/' + name)

    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('./history/' + name + '_history.csv', index=False)
