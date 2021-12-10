import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from custom_datagen import imageLoader
import yaml
import os

yfile = open("../settings2.yaml")
hp = yaml.load(yfile, Loader=yaml.FullLoader)

batch_size = 2  # Check IoU for a batch of images

# iou_result = {
#     "origin": 0.0,
#     "gauss": 0.0,
#     "s&p": 0.0,
#     "poisson": 0.0,
#     "speckle": 0.0
# }

iou_result=[]


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col = image.shape
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = image + image * gauss
        return noisy



SplitDataPath = hp.get("BraTS2020").get("SplitDataPath")

val_img_dir = os.path.join(SplitDataPath, "val", "images", "")
val_mask_dir = os.path.join(SplitDataPath, "val", "masks", "")

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

save_model_path = './models/model_3000_trick.hdf5'

# For predictions you do not need to compile the model, so ...
my_model = load_model(save_model_path,compile=False)

# Verify IoU on a batch of images from the test dataset
# Using built in keras function for IoU
# Only works on TF > 2.0
from keras.metrics import MeanIoU

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
# iou_result["origin"] = IOU_keras.result().numpy()
iou_result.append(IOU_keras.result().numpy())
for mode in ['gauss', 's&p', 'poisson', 'speckle']:
    noise_image_batch = []
    for n in range(batch_size):
        img = test_image_batch[n, :, :, :, :]
        img_noise = []
        for t in range(3):
            img_t = img[:, :, :, t]
            n_img = []
            for slice in range(128):
                noise_img = noisy(mode, img_t[:, :, slice])
                n_img.append(noise_img)
            n_img = np.array(n_img).transpose(2,1,0)
            img_noise.append(n_img)
        img_noise = np.array(img_noise).transpose(1,2,3,0)
        noise_image_batch.append(img_noise)
    noise_image_batch = np.array(noise_image_batch)

    test_pred_batch = my_model.predict(noise_image_batch)
    test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

    n_classes = 4
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
    # iou_result[mode] = IOU_keras.result().numpy()
    iou_result.append(IOU_keras.result().numpy())

# import json
# jsObj = json.dump(iou_result)
with open('error_epoch_100.txt','w') as file:
    for i in iou_result:
        file.write(str(i))
        file.write('\n')

plt.bar(('origin','gauss','s&p', 'poisson', 'speckle'),iou_result)
plt.savefig('./pictures/epoch_100_errors.png',dpi=600)
plt.show()