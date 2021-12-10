#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xinyu Li
Time    : 2021/12/2 16:14
Desc    : 
"""
import os
import numpy as np
from custom_datagen import imageLoader
import tensorflow as tf
from simple_3d_unet import simple_unet_model
from matplotlib import pyplot as plt
import segmentation_models_3D as sm
from keras.models import load_model
##############################################################
n_epochs = 1000
is_continue_train = True
load_model_path = './models/model_3000_trick.hdf5'
save_model_path = './models/model_5000_trick.hdf5'
##############################################################
# Define the image generators for training and validation
import yaml
yfile = open("../settings2.yaml")
hp = yaml.load(yfile,Loader=yaml.FullLoader)

SplitDataPath = hp.get("BraTS2020").get("SplitDataPath")
train_img_dir = os.path.join(SplitDataPath,"train","images","")
train_mask_dir = os.path.join(SplitDataPath,"train","masks","")

val_img_dir = os.path.join(SplitDataPath,"val","images","")
val_mask_dir = os.path.join(SplitDataPath,"val","masks","")

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)
##################################

########################################################################
batch_size = 2
train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader(val_img_dir, val_img_list,
                              val_mask_dir, val_mask_list, batch_size)

###########################################################################
# Define loss, metrics and optimizer to be used for training
wt0, wt1, wt2, wt3 = 0.26, 22.53, 22.53, 26.21
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
total_loss = dice_loss
metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]
LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)
#######################################################################
# Fit the model
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

if is_continue_train:
    # Load model for prediction or continue training

    # Now, let us add the iou_score function we used during our initial training
    model = load_model(load_model_path,
                          custom_objects={'dice_loss': dice_loss,
                                          'iou_score': sm.metrics.IOUScore(threshold=0.5)})
else:
    model = simple_unet_model(IMG_HEIGHT=128,
                              IMG_WIDTH=128,
                              IMG_DEPTH=128,
                              IMG_CHANNELS=3,
                              num_classes=4)

model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

log_dir = "./logs/epoch_100_trick"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)

history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=n_epochs,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch,
                    callbacks=[tensorboard_callback]
                    )

model.save(save_model_path)
##################################################################

# plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("./pictures/loss_{}".format(n_epochs),dpi=600)
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("./pictures/accuracy_{}".format(n_epochs),dpi=600)
plt.show()
#################################################

