from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

image_size = [224, 224]
data_path = './dip_model_data_balanced_clean/test'
# models_name = ['VGG 16 Unbalanced', 'VGG 16 Balanced']
models_name = [
    'vgg16',
    'vgg16_balanced',
    'vgg16_balanced_data_augmentation',
    'vgg16_balancedAll',
    'vgg16_unbalancedAll',
    'vgg16_balanced_fineTuning',
    'vgg16_balanced_crop_10',
    'vgg16_balanced_crop_20',
    'vgg16_balanced_crop_30',
    'vgg16_balanced_crop_40',
    'vgg16_balanced_crop_50',
    'vgg16_balanced_crop_60'
]



def load_test_data(path, size):
    gen_test = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = gen_test.flow_from_directory(
        path,
        target_size=(size, size),
        class_mode=None,
        shuffle=False
    )
    return test_generator


def load_models(type, models_name):
    models = []
    for name in models_name:
        model = load_model('./saved_model/' + type + '_model_' + name)
        models.append(model)
    return models


def predict(models, plot=True):
    index = 0

    for model in models:
        name = models_name[index]

        dataset = './dip_model_data'
        test_size = 0
        true = np.array([])

        if name == 'vgg16':
            dataset = './dip_model_data_unbalanced'
            true = np.array([0] * 933 + [1] * 269)
        elif name == 'vgg16_balancedAll':
            dataset = './dip_model_data_balancedAll'
            true = np.array([0] * 450 + [1] * 450)
        elif name == 'vgg16_unbalancedAll':
            dataset = './dip_model_data_unbalancedAll'
            true = np.array([0] * 3348 + [1] * 467)
        else:
            dataset = './dip_model_data_balanced'
            true = np.array([0] * 150 + [1] * 150)

        test_data = load_test_data(dataset + '/test', 180)

        probabilities = model.predict(test_data)
        prediction = probabilities > 0.5
        fpr, tpr, thresholds = roc_curve(true, probabilities)
        matrix = confusion_matrix(true, prediction)

        print(name + ":")
        print(matrix)
        print(roc_auc_score(true, probabilities))

        plt.plot(fpr, tpr)
        index += 1

    if plot:
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(models_name)
        plt.show()


def plot_history(path):
    df = pd.read_csv(path)
    plt.plot(df['accuracy'])
    plt.plot(df['val_accuracy'])
    plt.xlabel('# of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.show()


def plot_all_history(model_names, row, col):
    fig, axs = plt.subplots(row, col)
    index = 0
    for i in range(row):
        for j in range(col):
            name = model_names[index]

            df = pd.read_csv('./history/dip_model_' + name + '_history.csv')
            axs[i, j].plot(df['accuracy'])
            axs[i, j].plot(df['val_accuracy'])
            axs[i, j].set(xlabel='# of Epochs', ylabel='Accuracy')
            axs[i, j].legend(['accuracy', 'val_accuracy'])
            axs[i, j].set_title(name)
            axs[i, j].label_outer()

            index += 1
    plt.show()


# test_data = load_test_data(data_path, 180)
# models = loadModels('dip', ['vgg16', 'vgg16_balanced'])
models = load_models('dip', models_name)
predict(models)

# plot_all_history(models_name, 2, 2)

# plot_history('./history/dip_model_vgg16_balanced_history.csv')
# plot_history('./history/dip_model_vgg16_balanced_history.csv')
# plot_history('./history/dip_model_vgg16_balanced_data_augmentation_history.csv')

# true = np.array([0] * test_size + [1] * test_size)
# model = load_model('./saved_model/dip_model_vgg16_balanced')
# probabilities = model.predict(test_data)
# prediction = probabilities > 0.5
# fpr, tpr, thresholds = roc_curve(true, probabilities)
# print(confusion_matrix(true, prediction))
# print(roc_auc_score(true, probabilities))
