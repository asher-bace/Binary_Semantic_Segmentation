import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import unet  
from models import resnet34_unet
from oxford_pet import OxfordPets 
import utils
from utils import dice_score, iou_score, plot_training, count_parameters  # 輔助函數
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from glob import glob

# 超參數設定 #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 1
IMG_SIZE = (256, 256)
NUM_CLASSES = 2 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

# 載入資料集 #
DATASET_PATH = os.path.join("dataset", "oxford-iiit-pet")
DATASET_PATH = os.path.join(BASE_DIR, DATASET_PATH)
IMAGE_PATH = os.path.join(DATASET_PATH, "images")
MASK_PATH = os.path.join(DATASET_PATH, "annotations", "trimaps")

image_files = sorted(glob(os.path.join(IMAGE_PATH, "*.jpg")))
mask_files = sorted(glob(os.path.join(MASK_PATH, "*.png")))

train_img, val_test_img, train_mask, val_test_mask = train_test_split(image_files, mask_files, test_size=0.2, random_state=42)
val_img, test_img, val_mask, test_mask = train_test_split(val_test_img, val_test_mask, test_size=0.5, random_state=42)

# 資料增強 #
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

# 建立 Dataset & DataLoader #
train_dataset = OxfordPets(train_img, train_mask, transform=transform)
val_dataset = OxfordPets(val_img, val_mask, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型 #
unet_model = unet.UNet(input_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
resnet34_unet_model = resnet34_unet.ResNet34UNet(num_classes=NUM_CLASSES).to(DEVICE)
print(f"🔍 U-Net 模型總參數量: {count_parameters(unet_model)}")
print(f"🔍 ResNet & U-Net 模型總參數量: {count_parameters(resnet34_unet_model)}\n")

# 損失函數 & 優化器 #
criterion = nn.CrossEntropyLoss()  # 二元分類損失 (Binary Cross Entropy Loss)
unet_optimizer = optim.Adam(unet_model.parameters(), lr=LR)
resnet34_unet_optimizer = optim.Adam(resnet34_unet_model.parameters(), lr=LR)

# 訓練模型 #
unet_history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
unet_best_dice = 0.0
resnet34_unet_history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
resnet34_unet_best_dice = 0.0

for epoch in range(EPOCHS):
    print(f"\n📌 Epoch {epoch+1}/{EPOCHS}")

    # ----------- 訓練 U-Net -----------
    print("\n🧠 Training U-Net...")
    unet_train_loss, unet_train_dice, unet_train_iou = utils.train_one_epoch(
        unet_model, train_loader, unet_optimizer, criterion, DEVICE)
    unet_val_loss, unet_val_dice, unet_val_iou = utils.validate(
        unet_model, val_loader, criterion, DEVICE)

    unet_history["train_loss"].append(unet_train_loss)
    unet_history["val_loss"].append(unet_val_loss)
    unet_history["train_dice"].append(unet_train_dice)
    unet_history["val_dice"].append(unet_val_dice)

    print(f"[U-Net Train] Loss: {unet_train_loss:.4f}, Dice: {unet_train_dice:.4f}, IoU: {unet_train_iou:.4f}")
    print(f"[U-Net Val]   Loss: {unet_val_loss:.4f}, Dice: {unet_val_dice:.4f}, IoU: {unet_val_iou:.4f}")

    if unet_val_dice > unet_best_dice:
        unet_best_dice = unet_val_dice
        torch.save(unet_model.state_dict(), os.path.join(SAVE_DIR, "best_unet_model.pth"))
        print("✅ U-Net 最佳模型已更新！")

    # ----------- 訓練 ResNet34-UNet -----------
    print("\n🧠 Training ResNet34-UNet...")
    resnet_train_loss, resnet_train_dice, resnet_train_iou = utils.train_one_epoch(
        resnet34_unet_model, train_loader, resnet34_unet_optimizer, criterion, DEVICE)
    resnet_val_loss, resnet_val_dice, resnet_val_iou = utils.validate(
        resnet34_unet_model, val_loader, criterion, DEVICE)

    resnet34_unet_history["train_loss"].append(resnet_train_loss)
    resnet34_unet_history["val_loss"].append(resnet_val_loss)
    resnet34_unet_history["train_dice"].append(resnet_train_dice)
    resnet34_unet_history["val_dice"].append(resnet_val_dice)

    print(f"[ResNet34-UNet Train] Loss: {resnet_train_loss:.4f}, Dice: {resnet_train_dice:.4f}, IoU: {resnet_train_iou:.4f}")
    print(f"[ResNet34-UNet Val]   Loss: {resnet_val_loss:.4f}, Dice: {resnet_val_dice:.4f}, IoU: {resnet_val_iou:.4f}")

    if resnet_val_dice > resnet34_unet_best_dice:
        resnet34_unet_best_dice = resnet_val_dice
        torch.save(resnet34_unet_model.state_dict(), os.path.join(SAVE_DIR, "best_resnet34_unet_model.pth"))
        print("✅ ResNet34-UNet 最佳模型已更新！")

unet_history["train_loss"].append(float(unet_train_loss))
unet_history["val_loss"].append(float(unet_val_loss))
unet_history["train_dice"].append(float(unet_train_dice))
unet_history["val_dice"].append(float(unet_val_dice))

resnet34_unet_history["train_loss"].append(float(resnet_train_loss))
resnet34_unet_history["val_loss"].append(float(resnet_val_loss))
resnet34_unet_history["train_dice"].append(float(resnet_train_dice))
resnet34_unet_history["val_dice"].append(float(resnet_val_dice))
# 8️⃣ **繪製 Loss & Dice Score**
plot_training(unet_history)
plot_training(resnet34_unet_history)






