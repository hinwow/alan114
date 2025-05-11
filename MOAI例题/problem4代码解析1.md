



### 代码补全  
```python
import torch
import pandas as pd

test_images = torch.load('test_images.pt')
# 按照前面的方法归一化（假设小题1中是除以255进行归一化）
test_images = test_images / 255.0  
model.eval()
with torch.no_grad():
    test_images = test_images.to(device)
    outputs = model(test_images)
    predictions = outputs.argmax(dim=1)

df_test = pd.DataFrame({"label": predictions.cpu().numpy()})
df_test.to_csv("submission.csv", index_label="id")
```  

### 代码解释  
1. **归一化测试图像**：  
   `test_images = test_images / 255.0`，假设小题1中图像归一化是将像素值除以255（从 `[0, 255]` 缩放到 `[0, 1]`），此处对测试图像进行相同操作。  

2. **模型评估模式**：  
   `model.eval()`，将模型设为评估模式（如关闭 `Dropout`、使用 `BatchNorm` 训练时的全局统计量）。  

3. **禁用梯度计算**：  
   `with torch.no_grad()`，验证阶段无需计算梯度，节省内存并加速推理。  

4. **前向传播与预测**：  
   - `test_images = test_images.to(device)`，将测试图像移到模型所在设备（CPU 或 GPU）。  
   - `outputs = model(test_images)`，模型前向传播得到预测结果（未归一化概率）。  
   - `predictions = outputs.argmax(dim=1)`，对每个样本取概率最大的类别作为预测标签。  

5. **保存预测结果**：  
   - `df_test = pd.DataFrame({"label": predictions.cpu().numpy()})`，将预测结果转换为 `DataFrame`。  
   - `df_test.to_csv("submission.csv", index_label="id")`，保存为 `CSV` 文件，设置索引标签为 `id`。  

若要根据准确率 `x` 计算得分，需先通过测试集计算准确率，再代入对应公式：  
- 当 `90 ≤ x ≤ 100`：得分 `= 50 - 2 * (100 - x)`  
- 当 `80 ≤ x < 90`：得分 `= 30 - (90 - x)`  
- 当 `50 ≤ x < 80`：得分 `= 20 * (x - 50) / 30`  
- 当 `x < 50`：得分 `= 0` 。 