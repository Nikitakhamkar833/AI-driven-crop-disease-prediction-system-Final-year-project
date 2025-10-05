### importing the libraries ####
import tensorflow as tf #### for CNN 
import os
import numpy as np ### perform array operations
from keras_preprocessing.image import ImageDataGenerator  ### for data augmentation 
import matplotlib.pyplot as plt ### for visualization

###########  Train Dataset   ############
train_dir = os.path.join(r"F:\Projects (2024-25)\crop diseases system\dataset\Dataset\Train") #Train directory

############  Test Dataset   ############
test_dir = os.path.join(r"F:\Projects (2024-25)\crop disease system\dataset\Dataset\Test") # Test Directory


# # pre-processing
IMG_WIDTH = 224 #Width of image
IMG_HEIGHT = 224 # Height of Image

BATCH_SIZE = 32 # Batch Size 

train_data_size = 796
test_data_size = 88


train_data = ImageDataGenerator(
                rescale = 1./255, #normalizing the input image
                rotation_range = 60, # randomly rotate images in the range (degrees, 0 to 180)
                shear_range = 0.2,
                zoom_range = 0.3,
                horizontal_flip = True,
                brightness_range = (0.5, 1.5))

test_data = ImageDataGenerator(
                rescale = 1./255)

train_set = train_data.flow_from_directory(
                train_dir,
                target_size=(IMG_WIDTH,IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                class_mode = 'categorical')

test_set = test_data.flow_from_directory(
                test_dir,
                target_size = (IMG_WIDTH,IMG_HEIGHT),
                batch_size = BATCH_SIZE,
                shuffle=False,
                class_mode = 'categorical')

labels_values,no_of_images = np.unique(train_set.classes,return_counts = True)
dict(zip(train_set.class_indices,no_of_images))
labels = test_set.class_indices
labels = { v:k for k,v in labels.items() } # Flipping keys and values
values_lbl = list(labels.values()) # Taking out only values from dictionary
