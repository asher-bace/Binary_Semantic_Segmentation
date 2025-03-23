import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import unet  
from models import resnet34_unet
from oxford_pet import OxfordPets  
import utils
from utils import dice_score, iou_score  
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob

# 超參數設定 #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
IMG_SIZE = (256, 256)
NUM_CLASSES = 2  

# 載入測試集 #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join("dataset", "oxford-iiit-pet")
DATASET_PATH = os.path.join(BASE_DIR, DATASET_PATH)
IMAGE_PATH = os.path.join(DATASET_PATH, "images")
MASK_PATH = os.path.join(DATASET_PATH, "annotations", "trimaps")

test_img_files = sorted(glob(os.path.join(IMAGE_PATH, "*.jpg")))
test_mask_files = sorted(glob(os.path.join(MASK_PATH, "*.png")))

# 資料處理 #
transform = A.Compose([
    A.HorizontalFlip(p=0.5),                      # 50% 機率水平翻轉
    A.RandomRotate90(p=0.5),                      # 隨機旋轉 90 度
    A.Affine(                                     # 隨機平移、縮放、旋轉
        scale=(0.9, 1.1), 
        translate_percent=(0.0, 0.1), 
        rotate=(-15, 15), p=0.5
        ),           
    A.RandomBrightnessContrast(p=0.3),            # 隨機亮度對比
    A.GaussianBlur(p=0.2),                        # 模糊影像
    A.Normalize(mean=(0.5, 0.5, 0.5),             # Normalize for image
                std=(0.5, 0.5, 0.5)),
    ToTensorV2()                                  # 轉成 PyTorch Tensor
])


total_samples = len(test_img_files)
train_end = int(0.8 * total_samples)
val_end = int(0.9 * total_samples)  # 剩下的 20% 裡再分一半給 val / test

test_img = test_img_files[val_end:]
test_mask = test_mask_files[val_end:]

test_dataset = OxfordPets(test_img, test_mask, transform=None)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f"🧪 測試資料數量: {len(test_dataset)}")
print(f"📦 測試批次數量: {len(test_loader)}\n")

# 載入模型 #
unet_model = unet.UNet(input_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
resnet_model = resnet34_unet.ResNet34UNet(num_classes=NUM_CLASSES).to(DEVICE)

# ✅ 修正 `torch.load()`
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

# 執行評估**
criterion = nn.CrossEntropyLoss()
utils.evaluate(unet_model, test_loader, criterion, DEVICE, "U-Net")
utils.evaluate(resnet_model, test_loader, criterion, DEVICE, "ResNet34 & U-Net")







