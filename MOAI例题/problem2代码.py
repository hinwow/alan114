#小題 2：構建 CNN 模型 (15 分)
#定義一個簡單的 CNN 模型，滿足以下條件：
#2. 至少包含兩個池化層 ( nn.MaxPool2d ) (2分)
#3. 使用激活函數 ( nn.ReLU , nn.Sigmoid , nn.Tanh , nn.ELU , nn.LeakyReLU , nn.PreLU 等) (2分)
#4. 定義輸出層，輸出一個 10 維向量 (對應 10 個類別) (2分)
#5. 定義前向傳播，將以上所有網絡層連接 (7分)

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层 + 激活函数 + 池化层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 假设输入通道为1，可根据实际数据调整
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # 第二个卷积层 + 激活函数 + 池化层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        # 输出层（假设输入图像经两次池化后尺寸为7×7，如28×28输入）
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 32为第二个卷积层输出通道数，10对应10个类别
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # 展平张量
        x = self.fc(x)  # 输出10维向量
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
print(model)

#代码解析：​
#网络层定义（__init__方法）：​
#卷积层 1：nn.Conv2d(1, 16, kernel_size=3, padding=1)，输入通道为 1（可根据实际数据调整，如 RGB 图像输入通道为 3），输出通道 16，3×3 卷积核，padding=1保持输入输出尺寸一致。​
#激活函数 1：nn.ReLU()，引入非线性。​
#池化层 1：nn.MaxPool2d(2)，2×2 最大池化，降低空间维度。​
#卷积层 2：nn.Conv2d(16, 32, kernel_size=3, padding=1)，输入通道 16，输出通道 32。​
#激活函数 2：nn.ReLU()。​
#池化层 2：nn.MaxPool2d(2)。​
#输出层：nn.Linear(32 * 7 * 7, 10)，假设输入图像尺寸为 28×28（经两次池化后为 7×7），32 * 7 * 7为展平后的特征维度，输出 10 维向量（对应 10 个类别）。​
#前向传播（forward方法）：​
#依次执行卷积、激活、池化操作。​
#x.view(x.size(0), -1)将张量展平为一维向量，以便输入全连接层。​
#最后通过self.fc输出 10 维向量。​
#该模型满足题目中两个卷积层、两个池化层、使用激活函数（ReLU）、输出 10 维向量的要求，并正确定义了前向传播流程。
