### importing libraries
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


############  Train Dataset   ############
train_path  = os.path.join(r"F:\Projects (2022-23)\ disease\dataset\training") #Train directory

############  Test Dataset   ############
valid_path = os.path.join(r"F:\Projects (2022-23)\ disease\dataset\testing") # Test Directory

# re-size all the images to this
IMAGE_SIZE = [224, 224]

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False

# useful for getting number of classes
# folders = 5
  
# our layers - you can add more if you want
x = Flatten()(vgg.output)
prediction = Dense(4, activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


train_datagen = ImageDataGenerator(rescale = 1./255,
                rotation_range = 30, # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.2, # zooom the images
                brightness_range = (0.5, 1.5))

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# fit the model
history_model = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=100,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set))

model.save('trained_model_vgg16_updated_finalkjhgf.h5')

import matplotlib.pyplot as plt ### visualisation
plt.plot(history_model.history['accuracy'])
plt.plot(history_model.history['val_accuracy'])
plt.legend(['accuracy','val_accuracy'])
plt.show()

# Loss Comparison
plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()