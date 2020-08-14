import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_vnet3d import Vnet3dModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvimagedata = pd.read_csv('./dataprocess/train_img.csv')
    imagedata = csvimagedata.iloc[:, :].values
    csv_pos_data = pd.read_csv('./dataprocess/train_img_SE.csv')
    pos_data = csv_pos_data.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]

    Vnet3d = Vnet3dModule(256, 256, 16, channels=1, costname=("dice coefficient",))
    Vnet3d.train(imagedata, pos_data, "Vnet3d.pd", "log/diceVnet3d/", 0.001, 0.7, 10, 1)


train()
