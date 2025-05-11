#小題 1：數據讀取與預處理 (15 分)
# 讀取數據集圖片和標籤 (5分)
#將數據集按 8:2 的比例劃分為訓練集和驗證集。 (5分)
#將圖像數據歸一化並轉換為 Tensor。 (5分)
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split

# 读取数据集
images_raw = torch.load('train - images.pt')
labels_raw = pd.read_csv('train - labels.csv')

# 归一化数据集并转换为 torch.Tensor
images = torch.tensor(images_raw, dtype=torch.float32) / 255.0  # 假设图像数据是0 - 255范围，归一化到0 - 1
labels = torch.tensor(labels_raw.values, dtype=torch.long).squeeze()

# 创建数据集，并按照8:2划分成训练集和验证集
dataset = TensorDataset(images, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

