# 使用 U-Net 與 ResNet34 + U-Net 進行二元語義分割 (Binary Semantic Segmentation)

本專案實作了基於 **U-Net** 與 **ResNet34 + U-Net** 的二元語義分割模型，並應用於 [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) 來進行寵物區域的像素級分割任務，目標是區分「寵物（前景）」與「背景」。

## 🧠 專案簡介

語義分割 （Semantic Segmentation） 是將影像中的每個像素分類為特定類別的任務。在**二元語義分割**中，僅需分為兩類：前景（如目標物體）與背景。

本專案包含以下重點：
- 標準 **U-Net** 模型實作
- 使用 **ResNet34 為編碼器**、**U-Net 為解碼器** 的模型架構實作
- 以 **Oxford-IIIT Pet Dataset** 進行訓練、驗證與測試
- 評估指標（IoU、Dice Score）
- 模型儲存與最佳模型紀錄功能

## 📁 使用資料集

本專案使用的資料集為 [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)，該資料集包含 37 種貓狗品種的圖像與像素級標註。

在此任務中，我們將原始標註簡化為二元遮罩：
- 像素值為 `1`：表示寵物（前景）
- 像素值為 `0`：表示背景

## 🧩 模型架構簡介

### U-Net
U-Net 為對稱式的編碼器-解碼器 (Encoder-Decoder) 架構，搭配 Skip Connection，可有效保留高解析度特徵，廣泛應用於醫療影像與小型資料集語義分割任務。

### ResNet34 + U-Net
此架構使用 **ResNet34** 作為編碼器，有助於擷取更深層次特徵，並接上 U-Net 的解碼器部分以產生細緻分割結果。

## ⚙️ 環境安裝
下載 ZIP 或 Clone 專案
```bash
git clone https://github.com/asher-bace/Binary_Semantic_Segmentation.git
cd Binary_Semantic_Segmentation
pip install -r requirements.txt
```
## 📁 專案結構
```bash
.
├── dataset/
│   └── oxford-iiit-pet/
├── saved_models/
├── src/
│   ├── models/
│   │   ├── unet.py
│   │   └── resnet34_unet.py
│   ├── utils.py
│   ├── oxford_pet.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── requirements.txt
└── README.md

```
## 📦 資料集準備
請下載 Oxford-IIIT Pet Dataset，並將影像 (Images) 和標註檔案 (Annotations) 放至專案目錄中，如下所示：
```bash
.
├── dataset/
│    └── oxford-iiit-pet/
│       ├── images/
│       └── annotations/
```
## 🏃‍♂️ 模型訓練
調整 train.py 中的 EPOCHS 次數，之後輸入指令執行訓練：
```bash
cd src
python train.py
```
訓練完成後，最佳模型會保存在 /saved_models 目錄當中：
```bash
.
├── saved_models/
│    ├── best_unet_model.pth
│    └── best_resnet34_unet_model.pth
```
## 📊 結果展示
| Model | Loss|IoU (Val) |Dice Score (Val)|
|  :----:  | :----:  | :----:|:----:|
| U-Net  | 0.1307 |0.9246|0.9606|
| ResNet34 & U-Net  | 0.1460|0.9164 |0.9561|

U-Net Loss 與 Dice Score 折線圖：
![UNet](https://github.com/user-attachments/assets/e6365fce-bd7f-4df1-b3ef-4be3236e0ad6)

ResNet34 & U-Net Loss 與 Dice Score 折線圖：
![ResNet](https://github.com/user-attachments/assets/1bc83efe-b0ef-4233-b3e0-1b15c6e59dfb)

U-Net 測試結果：
![unet](https://github.com/user-attachments/assets/766ad611-e885-41bf-aab2-11a57df906e0)

ResNet34 & U-Net 測試結果：
![resnet](https://github.com/user-attachments/assets/76c66929-0302-4e2c-8131-60011c7c2a44)

## 🙏 感謝資源
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](<http://arxiv.org/abs/1505.04597>)
- [Deep Residual Learning for Image Recognition](<http://arxiv.org/abs/1512.03385>)
