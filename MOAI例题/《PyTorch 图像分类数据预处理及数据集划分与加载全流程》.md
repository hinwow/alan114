# 《PyTorch 图像分类数据预处理及数据集划分与加载全流程》

以下是代码的逐行详细解释，涵盖数据预处理、数据集创建、划分及加载的全流程：

### **一、图像数据处理：**`images = torch.tensor(images_raw, dtype=torch.float32) / 255.0`

#### **1. 代码功能**

将原始图像数据转换为 PyTorch 张量（`torch.Tensor`），并进行归一化处理，使其像素值范围从 `[0, 255]` 缩放到 `[0, 1]`。

#### **2. 关键步骤解析**

`torch.tensor(images_raw, dtype=torch.float32)`：

`images_raw` 是从 `train-images.pt` 读取的原始图像数据（可能是 NumPy 数组或其他格式）。

`torch.tensor()` 将其转换为 PyTorch 张量（深度学习模型只能处理张量数据）。

`dtype=torch.float32`：指定张量的数据类型为 32 位浮点数（模型计算常用类型，兼顾精度和内存）。

`/ 255.0`：

图像像素值通常以 8 位无符号整数（`uint8`）存储，范围是 `[0, 255]`。

归一化到 `[0, 1]` 可以避免因数值过大导致的梯度不稳定（例如，模型初始化权重较小，输入值过大会导致输出异常），同时加速模型收敛。

### **二、标签数据处理：**`labels = torch.tensor(labels_raw.values, dtype=torch.long).squeeze()`

#### **1. 代码功能**

将原始标签数据（CSV 文件）转换为 PyTorch 张量，并调整维度以适配模型输入要求。

#### **2. 关键步骤解析**

`labels_raw.values`：

`labels_raw` 是通过 `pd.read_csv('train-labels.csv')` 读取的 pandas DataFrame。

`.values` 提取 DataFrame 中的数值部分（返回 NumPy 数组），形状通常为 `(N, 1)`（N 为样本数，1 列为标签）。

`torch.tensor(..., dtype=torch.long)`：

将 NumPy 数组转换为 PyTorch 张量。

`dtype=torch.long`（长整型）：分类任务中，标签通常是整数索引（如 0/1/2 表示类别），需要用长整型存储。

`.squeeze()`：

去除张量中维度大小为 1 的维度。例如，原始标签形状为 `(N, 1)`，`squeeze()` 后变为 `(N,)`（一维张量），与模型期望的标签形状（`(batch_size,)`）一致。

### **三、创建数据集：**`dataset = TensorDataset(images, labels)`

#### **1. 代码功能**

将图像和标签张量组合成一个完整的数据集对象，便于后续划分和加载。

#### **2. 关键解析**

`TensorDataset`：

PyTorch 内置的数据集类，用于将多个张量（如特征和标签）打包成一个可迭代的数据集。

每个样本通过索引 `i` 访问，返回 `(images[i], labels[i])`，其中 `images[i]` 是第 i 张图像的张量，`labels[i]` 是第 i 张图像的标签。

### **四、划分训练集与验证集：**`train_dataset, val_dataset = random_split(dataset, [train_size, val_size])`

#### **1. 代码功能**

将完整的数据集按 8:2 的比例随机划分为训练集（Training Set）和验证集（Validation Set）。

#### **2. 关键步骤解析**

`train_size = int(0.8 * len(dataset))`：

计算训练集大小（占总样本的 80%）。例如，总样本数为 1000，则训练集大小为 800。

`val_size = len(dataset) - train_size`：

验证集大小为总样本数减去训练集大小（剩余 20%）。

`random_split(dataset, [train_size, val_size])`：

`random_split` 是 PyTorch 的数据集划分工具，按指定大小随机划分数据集。

**随机性**：划分时会随机打乱数据顺序（类似 `shuffle`），确保训练集和验证集的样本分布一致。

**注意**：若需复现划分结果，需通过 `generator=torch.Generator().manual_seed(seed)` 指定随机种子（用户代码中未显式设置）。

### **五、创建数据加载器：**`train_loader = DataLoader(...)`** 和 **`val_loader = DataLoader(...)`

#### **1. 代码功能**

将训练集和验证集包装为数据加载器（`DataLoader`），支持批量（Batch）加载数据、自动打乱（Shuffle）等功能。

#### **2. 关键参数解析**

`batch_size=128`：

每次从数据集中加载的样本数量（批量大小）。例如，训练集有 800 个样本，`batch_size=128` 会生成 7 个批次（最后一个批次可能不足 128）。

选择依据：受 GPU 内存限制（批量越大，内存占用越高），通常取 32、64、128 等 2 的幂次。

`shuffle=True`**（仅训练集）**：

训练时每次迭代前打乱数据顺序，避免模型因数据顺序的规律性（如同一类别的样本集中出现）而过拟合。

验证集通常设置 `shuffle=False`（用户代码中验证集未设置，默认 `False`），因为验证需要稳定的顺序以评估模型真实性能。

### **六、完整流程总结**



| 步骤         | 核心操作                                            | 目的                          |
| ---------- | ----------------------------------------------- | --------------------------- |
| 图像归一化      | `images = ... / 255.0`                          | 缩像素值范围，稳定模型训练               |
| 标签处理       | `labels = ... .squeeze()`                       | 转换为模型需要的长整型张量，并调整维度         |
| 创建数据集      | `TensorDataset(images, labels)`                 | 组合图像和标签，便于统一管理              |
| 划分训练 / 验证集 | `random_split(dataset, [train_size, val_size])` | 随机划分数据，评估模型泛化能力             |
| 创建数据加载器    | `DataLoader(..., batch_size=128, shuffle=True)` | 批量加载数据，支持并行读取（默认多进程），提升训练效率 |

### **潜在注意事项**

**数据类型匹配**：

图像通常用 `float32`（模型计算），标签用 `long`（分类任务的索引）。若标签是回归任务（如连续值），需用 `float32`。

**随机种子控制**：

若需复现实验结果，划分数据集时应设置随机种子：



```
generator = torch.Generator().manual\_seed(42)  # 固定随机种子

train\_dataset, val\_dataset = random\_split(dataset, \[train\_size, val\_size], generator=generator)
```

**验证集的作用**：

验证集用于在训练过程中评估模型性能（如调整超参数），不能参与模型训练（否则会导致 “验证集泄露”，评估结果不可靠）。