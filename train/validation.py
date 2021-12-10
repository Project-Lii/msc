from keras.models import load_model
import numpy as np
from custom_datagen import imageLoader
import yaml
import os
yfile = open("../settings2.yaml")
hp = yaml.load(yfile,Loader=yaml.FullLoader)

SplitDataPath = hp.get("BraTS2020").get("SplitDataPath")

val_img_dir = os.path.join(SplitDataPath,"val","images","")
val_mask_dir = os.path.join(SplitDataPath,"val","masks","")

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

save_model_path = './models/model_3000_trick.hdf5'

# For predictions you do not need to compile the model, so ...
my_model = load_model(save_model_path,compile=False)

# Verify IoU on a batch of images from the test dataset
# Using built in keras function for IoU
# Only works on TF > 2.0
from keras.metrics import MeanIoU

batch_size = 8  # Check IoU for a batch of images
test_img_datagen = imageLoader(val_img_dir, val_img_list,
                               val_mask_dir, val_mask_list, batch_size)

# Verify generator.... In python 3 next() is renamed as __next__()
test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("mean iou is {}.".format(IOU_keras.result().numpy()))


#############################################
# Predict on a few test images, one at a time
# Try images:
img_num = 1

test_img = np.load(os.path.join(val_img_dir,"image_" + str(img_num) + ".npy"))
test_mask = np.load(os.path.join(val_mask_dir,"mask_" + str(img_num) + ".npy"))
test_mask_argmax = np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

# print(test_prediction_argmax.shape)
# print(test_mask_argmax.shape)
# print(np.unique(test_prediction_argmax))


# Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
import random

# n_slice=random.randint(0, test_prediction_argmax.shape[2])
n_slice = 1
def show_one_testpic(n_slice):
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(test_mask_argmax[:, :, n_slice])
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction_argmax[:, :, n_slice])
    plt.show()
for i in range(59,70):
    show_one_testpic(i)
############################################################