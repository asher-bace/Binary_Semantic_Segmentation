import os
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models import unet  # 載入 U-Net 模型
from models import resnet34_unet
import albumentations as A
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import PIL
from PIL import ImageOps
import utils

# 設定參數 #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (256, 256)
NUM_CLASSES = 2 

# 載入測試集 #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join("dataset", "oxford-iiit-pet")
DATASET_PATH = os.path.join(BASE_DIR, DATASET_PATH)
IMAGE_PATH = os.path.join(DATASET_PATH, "images")
MASK_PATH = os.path.join(DATASET_PATH, "annotations", "trimaps")

# 讀取已訓練模型 #
unet_model = unet.UNet(input_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
resnet_model = resnet34_unet.ResNet34UNet(num_classes=NUM_CLASSES).to(DEVICE)

# 修正 torch.load 的警告 #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
unet_path = os.path.join(MODEL_DIR, "best_unet_model.pth")
resnet_path = os.path.join(MODEL_DIR, "best_resnet34_unet_model.pth")

unet_path = os.path.join(MODEL_DIR, "best_unet_model.pth")
resnet_path = os.path.join(MODEL_DIR, "best_resnet34_unet_model.pth")
unet_model.load_state_dict(torch.load(unet_path, weights_only=True))
resnet_model.load_state_dict(torch.load(resnet_path, weights_only=True))
unet_model.eval() 
resnet_model.eval()

# 資料處理 #
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


# 隨機抽取影像並執行推論 #
try:
    test_image_path = utils.get_random_image(IMAGE_PATH)
    print(f"📸 測試影像: {test_image_path}")
    utils.infer_image(test_image_path, MASK_PATH, transform, DEVICE, unet_model)
    utils.infer_image(test_image_path, MASK_PATH, transform, DEVICE, resnet_model)
except Exception as e:
    print(f"❌ 發生錯誤: {e}")