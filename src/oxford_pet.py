import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.model_selection import train_test_split
import utils

class OxfordPets(Dataset):
    def __init__(self, input_image_paths, target_image_paths, image_size = (256, 256), transform = None):
        self.input_image_paths = input_image_paths
        self.target_image_paths = target_image_paths
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.target_image_paths)
    
    def __getitem__(self, index):
        try:
            image_path = self.input_image_paths[index]
            mask_path = self.target_image_paths[index]

            # 將 "\" 換成 "/"，以避免讀取失敗的問題 #
            #image_path = image_path.replace("\\", "/")
            #mask_path = mask_path.replace("\\", "/")

            # 嘗試讀取影像 #
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"❌ 無法讀取影像: {image_path}")

            # 將顏色格式改為 RGB 並將影像大小調整為 256x256 #
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size, interpolation = cv2.INTER_NEAREST)

            # 嘗試讀取 Mask #
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"❌ 無法讀取 Mask: {mask_path}")
            
            # 調整 Mask 大小為 256x256 #
            mask = cv2.resize(mask, self.image_size, interpolation = cv2.INTER_LINEAR)

            # 轉換 Mask (1, 2, 3 -> 0, 1) #
            mask = np.where(mask == 1, 0, mask)
            mask = np.where(mask >= 2, 1, mask)

            # 影像增強 #
            self.transform = A.Compose([
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

            if self.transform:
                augmented = self.transform(image = image, mask = mask)
                image, mask = augmented["image"], augmented["mask"]

            # 轉換為 Pytorch Tensor #
            if isinstance(image, np.ndarray):
                image = torch.tensor(image, dtype = torch.float32).permute(2, 0, 1) / 255.0

            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype = torch.long)
            else:
                mask = mask.clone().detach().long()

            return image, mask
        
        except Exception as e:
            print(f"⚠️ 跳過樣本 (index {index}): {e}")

            return self.__getitem__((index + 1) % len(self))
        
if __name__ == "__main__":
    # 設定資料集路徑 #
    DATASET_PATH = "dataset/oxford-iiit-pet/"
    IMAGE_PATH = os.path.join(DATASET_PATH, "images")
    MASK_PATH = os.path.join(DATASET_PATH, "annotations/trimaps")

    # 讀取所有 images 和 annotations #
    image_files = sorted(glob(os.path.join(IMAGE_PATH, "*.jpg"), recursive=True))
    mask_files = sorted(glob(os.path.join(MASK_PATH, "*.png"), recursive=True))

    # 確保 images 和 annotations 數量一致 #
    assert len(image_files) == len(mask_files), "影像與標註數量不匹配！"

    # 隨機劃分資料集 (80% 訓練、10% 驗證、10% 測試) #
    train_img, temp_img, train_mask, temp_mask = train_test_split(image_files, mask_files, test_size=0.2, random_state=42)
    val_img, test_img, val_mask, test_mask = train_test_split(temp_img, temp_mask, test_size=0.5, random_state=42)

    # 建立 PyTorch Dataset
    train_dataset = OxfordPets(train_img, train_mask, transform=None)
    val_dataset = OxfordPets(val_img, val_mask, transform=None)
    test_dataset = OxfordPets(test_img, test_mask, transform=None)

    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 測試訓練集、驗證集、測試集的影像數量 #
    print(f"🔢 訓練資料數量: {len(train_dataset)}")
    print(f"🔍 驗證資料數量: {len(val_dataset)}")
    print(f"🧪 測試資料數量: {len(test_dataset)}\n")

    # 查看各個 DataLoader 的批次數量 #
    print(f"📦 訓練批次數量: {len(train_loader)}")
    print(f"📦 驗證批次數量: {len(val_loader)}")
    print(f"📦 測試批次數量: {len(test_loader)}\n")

    # 測試讀取訓練集的一個 batch #
    sample_images, sample_masks = next(iter(train_loader))

    print(f"影像批次大小: {sample_images.shape}")
    print(f"標註批次大小: {sample_masks.shape}\n")
    
    image_np = sample_images[0].permute(1, 2, 0).numpy()  # [C, H, W] → [H, W, C]
    mask_np = sample_masks[0].squeeze(0).numpy()  # [1, H, W] → [H, W]

    # 顯示影像與 Mask #
    utils.plot_batch_sample(image_np, mask_np)