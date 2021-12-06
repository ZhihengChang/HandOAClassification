"""
This script is for data pre-processing.
NOTE: to clean the data before data organization and other data processing method
1. organize the data: group the data based on the file name (finger joint): dip, pip, mcp and KL value
2. split data folder into train, validation and test (balanced or unbalanced)
3. balanced the data equal to number of cases in the minority class (optional)
3. center crop the data (optional)

Example usage:

*** Data organization ***
organizeData('FingerJoints', 'data/dip', 'dip', df)

*** Balancing the data ***
balance_data('./data_all')

*** Split data into train, val, and test folders ***

* for balance dataset:
    splitfolders.fixed( "./dip", output="dip_model_data_balanced",
        seed=1337, fixed=(150, 150),
        oversample=False,
        group_prefix=None
    )

* for unbalanced dataset:
    splitfolders.ratio("./DIP_unbalancedAll", output="output",
        seed=1337,
        ratio=(.8, .1, .1),
        group_prefix=None
    )

*** Crop the data ***
crop_center_on_dataset('./dip_model_data_balenced', 80)

"""

import os
import shutil
import cv2
import pandas as pd
import random
from glob import glob

# reads in the csv file as database
df = pd.read_csv('../hand.csv')
df = df.drop_duplicates(subset=['id'])


'''
organize the data based on the string_to_match in the source_folder and output into a dest_folder
NOTE: after calling this method, the source folder will be empty or left with data not found in the data base: df
'''
def organizeData(source_folder, dest_folder, string_to_match, df):
    # Check all files in source_folder
    try:
        for filename in os.listdir(source_folder):
            # Move the file if the filename contains the string to match
            if string_to_match in filename:
                # Check kl score
                name = filename.split('_')
                row = df.loc[df['id'] == int(name[0])]['v00' + name[1][0:4].upper() + '_KL']
                if(not row.isnull().bool()):
                    if(row.astype(int).item() <= 1):
                        shutil.move(os.path.join(source_folder, filename), dest_folder + '/noa')
                    else:
                        shutil.move(os.path.join(source_folder, filename), dest_folder + '/oa')
    except ValueError as e:
        print('At Filename: ' + filename)
        print(e)


'''
Balance the given dataset noa cases equal to oa cases
A out folder will be created contain all the random selected files from noa, number equal to the oa cases
NOTE: manually rename out folder to noa and remove the original noa folder
'''
def balance_data(dataset):
    n = len(glob(dataset + '/oa/*'))
    print(n)
    dir = dataset + '/noa'
    dst = dataset + '/out'

    files = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]

    if len(files) < n:
        for file in files:
            shutil.copyfile(os.path.join(dir, file), dst)
    else:
        for x in range(n):
            if len(files) == 0:
                break
            else:
                selection = random.randint(0, len(files) - 1)
                file = files.pop(selection)
                shutil.copy(os.path.join(dir, file), dst)


'''
Center crop all the data in the data_source_folder to the given size into the output_dest_folder
NOTE: e.g. crop 10 px on each side, the size would be (image width - 2 * 10)
'''
def crop_center(data_source_folder, output_dest_folder, size):
    # Check all files in source_folder
    for filename in os.listdir(data_source_folder):
        img = cv2.imread(os.path.join(data_source_folder, filename))
        print(filename)
        # crop center
        height, width = img.shape[0], img.shape[1]
        x = (width - size + 1) // 2
        y = (height - size + 1) // 2
        img = img[y:(y + size), x:(x + size), :]

        cv2.imwrite(os.path.join(output_dest_folder, filename), img)


'''
Center crop a given dataset: train, val, and test
'''
def crop_center_on_dataset(dataset, size):
    crop_center(dataset + '/train/noa', './dip_model_data_crop_' + str(size) + '/train/noa', size)
    crop_center(dataset + '/train/oa', './dip_model_data_crop_' + str(size) + '/train/oa', size)
    crop_center(dataset + '/val/noa', './dip_model_data_crop_' + str(size) + '/val/noa', size)
    crop_center(dataset + '/val/oa', './dip_model_data_crop_' + str(size) + '/val/oa', size)
    crop_center(dataset + '/test/noa', './dip_model_data_crop_' + str(size) + '/test/noa', size)
    crop_center(dataset + '/test/oa', './dip_model_data_crop_' + str(size) + '/test/oa', size)



# organizeData('FingerJoints', 'data/dip', 'dip', df)
# organizeData('FingerJoints', 'data/pip', 'pip', df)
# organizeData('FingerJoints', 'data/mcp', 'mcp', df)

# splitfolders.fixed("./dip", output="dip_model_data_balanced",
#   seed=1337, fixed=(150, 150),
#   oversample=False,
#   group_prefix=None
# )
# splitfolders.ratio("./DIP_unbalancedAll", output="output",
#   seed=1337,
#   ratio=(.8, .1, .1),
#   group_prefix=None
# )

# crop_center_on_dataset('./dip_model_data_balenced', 80)



