import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import cv2
import random
import numpy as np

# 繪製訓練集測試樣本 #
def plot_batch_sample(image, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Segmentation Mask")
    plt.axis("off")

    plt.show()

# 計算 Dice Score #
def dice_score(pred, target, epsilon=1e-6):
    # print(f"🔍 dice_score - Before: pred.shape: {pred.shape}, target.shape: {target.shape}")

    # **確保 pred 和 target 形狀相同**
    if pred.shape != target.shape:
        if pred.dim() == 4:  # [N, C, H, W] -> [N, H, W]
            pred = torch.argmax(pred, dim=1)  # 取最大類別，轉為 [N, H, W]
        pred = pred.float()  # 確保是 float

    target = target.float()  # 確保是 float

    # print(f"🔍 dice_score - After: pred.shape: {pred.shape}, target.shape: {target.shape}")

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice

# 計算 IoU (Intersection over Union) #
def iou_score(pred, target, smooth=1e-6):
    """
    計算 IoU Score
    - pred: (batch_size, num_classes, H, W) 或 (batch_size, H, W)
    - target: (batch_size, H, W)
    """
    if pred.dim() == 4:  # 如果 `pred` 還有 num_classes 維度，則轉換
        pred = torch.argmax(pred, dim=1)

    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# 記錄並可視化訓練過程 (Loss & Accuracy) #
def plot_training(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # ✅ 確保 Tensor 轉換為 CPU 並轉換為 NumPy
    train_dice = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in history["train_dice"]]
    val_dice = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in history["val_dice"]]

    train_loss = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in history["train_loss"]]
    val_loss = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in history["val_loss"]]

    # ✅ 繪製 Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")

    # ✅ 繪製 Dice Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_dice, label="Train Dice")
    plt.plot(epochs, val_dice, label="Val Dice")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.title("Training & Validation Dice Score")

    plt.show()

# 計算模型參數數量 #
def count_parameters(model):
    """
    計算模型的總參數數量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 訓練函數 #
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0

    loop = tqdm(train_loader, desc="Training", leave=True)
    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)

        # print(f"🔍 outputs={outputs.shape}, dtype={outputs.dtype}")

        loss = criterion(outputs, masks)  # `outputs`: float, `masks`: long
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        dice_total += dice_score(outputs, masks)
        iou_total += iou_score(outputs, masks)

        loop.set_postfix(loss=loss.item())

    return running_loss / len(train_loader), dice_total / len(train_loader), iou_total / len(train_loader)

# 驗證函數 #
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0

    with torch.no_grad():
        loop = tqdm(val_loader, desc="Validating", leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            dice_total += dice_score(outputs, masks)
            iou_total += iou_score(outputs, masks)

            loop.set_postfix(val_loss=loss.item())

    return running_loss / len(val_loader), dice_total / len(val_loader), iou_total / len(val_loader)


# 評估函數 #
def evaluate(model, test_loader, criterion, device, model_name):
    model.eval()
    running_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating", leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)  # 取最大類別
            masks = masks.squeeze(1)  # 確保 masks 也是單通道

            # ✅ 轉換 `dtype`，避免錯誤
            preds = preds.float()
            masks = masks.float()

            # **Debug log**
            #print(f"✅ preds.shape: {preds.shape}, masks.shape: {masks.shape}")
            #print(f"✅ preds.dtype: {preds.dtype}, masks.dtype: {masks.dtype}")

            dice_total += dice_score(preds, masks)
            iou_total += iou_score(preds, masks)
            running_loss += loss.item()

            loop.set_postfix(val_loss=loss.item())

    avg_loss = running_loss / len(test_loader)
    avg_dice = dice_total / len(test_loader)
    avg_iou = iou_total / len(test_loader)

    print(f"\n✅ {model_name} 評估完成！")
    print(f"📉 Loss: {avg_loss:.4f}")
    print(f"🎯 Dice Score: {avg_dice:.4f}")
    print(f"📊 IoU Score: {avg_iou:.4f}")

    return avg_loss, avg_dice, avg_iou

# **4️⃣ 隨機選取一張影像**
def get_random_image(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        raise FileNotFoundError(f"❌ 影像資料夾 {image_dir} 沒有可用的影像！")
    return os.path.join(image_dir, random.choice(image_files))

# **5️⃣ 取得 Ground Truth Mask 路徑**
def get_mask_path(image_path, MASK_DIR):
    filename = os.path.basename(image_path)
    mask_filename = filename.replace(".jpg", ".png").replace(".jpeg", ".png")
    mask_path = os.path.join(MASK_DIR, mask_filename)
    return mask_path if os.path.exists(mask_path) else None

# **6️⃣ 推論影像**
def infer_image(image_path, mask_path, transform, DEVICE, model):
    # ✅ **檢查影像檔案是否存在**
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 找不到影像檔案: {image_path}")

    # **讀取影像**
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ 無法讀取影像: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 轉換為 RGB 格式
    original_size = image.shape[:2]  # 原始影像大小 (H, W)

    # **前處理**
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)  # [1, 3, 256, 256]

    # **執行推論**
    with torch.no_grad():
        output = model(image_tensor)  # [1, num_classes, 256, 256]
        preds = torch.argmax(output, dim=1)  # 取最大類別
        preds = preds.squeeze(0).cpu().numpy()  # [256, 256]

    # ✅ **確保輸出解析度與原圖相同**
    preds_resized = cv2.resize(preds.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    # **讀取 Ground Truth Mask**
    mask_path = get_mask_path(image_path, mask_path)
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        mask_resized = torch.tensor(mask_resized, dtype=torch.float32)
        mask_resized = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8) * 255  # Torch 方式自動對比增強
        mask_resized = mask_resized.numpy().astype(np.uint8)
    else:
        mask_resized = np.zeros(original_size, dtype=np.uint8)  # 若無 mask，則顯示全黑

    # **視覺化結果**
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_resized, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(preds_resized, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.show()