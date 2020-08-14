import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from ResUnet.model_resnet import ResUnet3dModule
import numpy as np
import pandas as pd
import cv2
import os


def nor_data(images):
    max = np.max(images)
    min = np.min(images)
    new_images = (images-min)/(max-min)
    return new_images


def predict():
    height = 512
    width = 512
    dimension = 16
    ResUnet3d = ResUnet3dModule(height, width, dimension, channels=1, inner_channel=16, costname=("mse",),
                                inference=True, model_path="log/mseResUnet3d/model/ResUnet3d.pd")
    src_image_path = r'/mnt/02520c27-ec8e-4661-b88f-05aa2011ffa7/lhk/denoise/test'
    predict_save_path = ""
    # 得到test_csv里各个病人的z范围
    csv_pos_data = pd.read_csv('./dataprocess/test_img_SE.csv')
    pos_data = csv_pos_data.iloc[:, :].values
    for num in pos_data[0]:
        LD_file_path = os.path.join(src_image_path, num, 'LDCT')
        HD_file_path = os.path.join(src_image_path, num, 'HDCT')
        start = pos_data[1]
        end = pos_data[2]
        # 读取数据
        LD_list = os.listdir(LD_file_path)
        length = len(LD_list)
        LD_images = np.zeros((length, 512, 512), dtype=np.float32)
        HD_images = np.zeros((length, 512, 512), dtype=np.float32)
        for i in range(length):
            slice = int(LD_list[i].split('_')[2].split('.')[0]) - 1
            img = np.fromfile(LD_file_path + '/' + LD_list[i], dtype='float32')
            img = img.reshape((512, 512))
            LD_images[slice, :, :] = img
            HD_img = np.fromfile(HD_file_path + '/' + LD_list[i], dtype='float32')
            HD_img = HD_img.reshape((512, 512))
            HD_images[slice, :, :] = HD_img
        LD_images_part = LD_images[start:end, :, :]
        HD_images_part = nor_data(HD_images[start:end, :, :])
        HD_images_part = HD_images_part * 255
        length = end-start+1
        LD_images_part = np.reshape(LD_images_part, (length, height, width, 1))
        output = np.zeros((index, height, width), np.float32)

        for i in range(0, length + dimension, dimension // 2):
            if (i + dimension) <= length:
                image_data = LD_images_part[i:i + dimension, :, :, :]
                output[i:i + dimension, :, :] = ResUnet3d.prediction(image_data)
            elif (i < length):
                image_data = LD_images_part[length - dimension:length, :, :, :]
                output[length - dimension:index, :, :] = ResUnet3d.prediction(image_data)

        mask = output.copy()
        result = np.clip(mask, 0, 255).astype('uint8')
        HD_result = np.clip(HD_images_part, 0, 255).astype('uint8')
        for i in range(0, length):
            cv2.imwrite(predict_save_path + "/LD/" + str(i) + ".bmp", result[i])
            cv2.imwrite(predict_save_path + "/HD/" + str(i) + ".bmp", HD_result[i])


predict()
