"""
Model training script
"""
from glob import glob
import numpy as np
from src import DataProcessor, Models

'''
Set path and model parameters 
'''
dataset = './dip_model_data_balanced_clean'
train_path = dataset + '/train'
test_path = dataset + '/val'

BATCH = 128
EPOCHS = 300
image_size = (180, 180)

image_files = glob(train_path + '/*/*.png')
test_image_files = glob(test_path + '/*/*.png')
train_step = int(np.ceil(len(image_files)/BATCH))
val_step = int(np.ceil(len(test_image_files)/BATCH))

generator = DataProcessor.DEFAULT_GENERATOR
adv_generator = DataProcessor.DATA_AUGMENTATION_GENERATOR

# load unbalanced data
train_data_unbalanced = DataProcessor.load_train_data('./dip_model_data_unbalanced', generator, image_size)
val_data_unbalanced = DataProcessor.load_val_data('./dip_model_data_unbalanced', generator, image_size)

# load balanced data
train_data = DataProcessor.load_train_data('./dip_model_data_balanced', generator, image_size)
val_data = DataProcessor.load_val_data('./dip_model_data_balanced', generator, image_size)

'''
Train vgg 16 using balanced data set, no center crop
'''
vgg16_balanced = Models.vgg_pretrained((180, 180, 3), 0)
Models.train_save(vgg16_balanced, 'dip_model_vgg16_balanced',
                  train_data, val_data, EPOCHS, BATCH,
                  int(np.ceil(4756/BATCH)), int(np.ceil(300/BATCH)))

# load balanced data and apply data augmentation
train_data_augmentation = DataProcessor.load_train_data('./dip_model_data_balanced', adv_generator, image_size)
val_data_augmentation = DataProcessor.load_val_data('./dip_model_data_balanced', adv_generator, image_size)

'''
Train vgg 16 using balanced data set, no center crop, with data augmentation
'''
vgg16_balanced_data_augmentation = Models.vgg_pretrained((180, 180, 3), 0)
Models.train_save(vgg16_balanced_data_augmentation, 'dip_model_vgg16_balanced_data_augmentation',
                  train_data_augmentation, val_data_augmentation, EPOCHS, BATCH,
                  int(np.ceil(4756/BATCH)), int(np.ceil(300/BATCH)))

'''
Train vgg 16 with fine-tuning using balanced data set, no center crop
'''
vgg16_balanced_fineTuning = Models.vgg_pretrained((180, 180, 3), 0, True)
Models.train_save(vgg16_balanced_fineTuning, 'dip_model_vgg16_balanced_fineTuning',
                  train_data, val_data, EPOCHS, BATCH,
                  int(np.ceil(4756/BATCH)), int(np.ceil(300/BATCH)))


#load balanced data including all joints
train_data_all = DataProcessor.load_train_data('./dip_model_data_balancedAll', generator, image_size)
val_data_all = DataProcessor.load_val_data('./dip_model_data_balancedAll', generator, image_size)

'''
Train vgg 16 with fine-tuning using balanced_all data set, no center crop
'''
vgg16_balanced_all = Models.vgg_pretrained((180, 180, 3), 0)
Models.train_save(vgg16_balanced_all, 'dip_model_vgg16_balancedAll',
                  train_data_all, val_data_all, EPOCHS, BATCH,
                  int(np.ceil(7128/BATCH)), int(np.ceil(900/BATCH)))


#load unbalanced data including all joints
train_data_unbalancedAll = DataProcessor.load_train_data('./dip_model_data_unbalancedAll', generator, image_size)
val_data_unbalancedAll = DataProcessor.load_val_data('./dip_model_data_unbalancedAll', generator, image_size)

'''
Train vgg 16 with fine-tuning using unbalanced_all data set, no center crop
'''
vgg16_unbalanced_all = Models.vgg_pretrained((180, 180, 3), 0)
Models.train_save(vgg16_unbalanced_all, 'dip_model_vgg16_unbalancedAll',
                  train_data_unbalancedAll, val_data_unbalancedAll, EPOCHS, BATCH,
                  int(np.ceil(30506/BATCH)), int(np.ceil(3812/BATCH)))

'''
Train vgg 16 using balanced data set, with different center crop
'''
for crop in [10, 20, 30, 40, 50, 60]:

    model = Models.vgg_pretrained((180, 180, 3), 10)
    Models.train_save(model, 'dip_model_vgg16_balanced_crop_' + str(crop),
                      train_data, val_data, EPOCHS, train_step, val_step)

