# 逻辑回归（Logistic Regression）详解  

逻辑回归（Logistic Regression）是一种经典的**二分类算法**（也可扩展至多分类），尽管名字含“回归”，但实际用于解决分类问题。它通过线性回归结合Sigmoid函数，将输出映射到概率区间，从而实现分类。本文从原理到代码，全面解析逻辑回归。


## **一、核心概念：为什么逻辑回归是分类算法？**  
逻辑回归的本质是**概率分类模型**：  
- **输入**：特征 \( x \)（如肿瘤大小、年龄）。  
- **输出**：样本属于正类（标签1）的概率 \( P(y=1 \mid x) \)（取值范围 \([0,1]\)）。  
- **决策**：设定阈值（默认0.5），概率≥0.5则分类为1，否则为0。  


## **二、核心原理：从线性回归到逻辑回归**  
### 1. 线性回归的局限性  
线性回归的输出 \( z = w^T x + b \) 是连续值，无法直接用于分类（如0/1标签）。  


### 2. Sigmoid函数：将线性输出映射到概率  
逻辑回归通过 **Sigmoid函数** 将线性输出 \( z \) 压缩到 \([0,1]\) 区间，公式为：  
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$  
其中 \( z = w^T x + b \)（线性组合）。  


### 3. 模型输出概率  
逻辑回归的输出是样本属于正类的概率：  
$$ P(y=1 \mid x) = \sigma(w^T x + b) $$  


### 4. 决策边界  
通过设定阈值（如0.5），将概率转换为类别：  
- 若 \( P(y=1 \mid x) \geq 0.5 \)，预测为正类（1）。  
- 若 \( P(y=1 \mid x) < 0.5 \)，预测为负类（0）。  


## **三、数学公式：损失函数与优化**  
### 1. 损失函数（交叉熵损失）  
逻辑回归使用**交叉熵损失**（对数损失）作为优化目标，公式为：  
$$ L(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$  
其中 \( y_i \) 是真实标签（0或1），\( \hat{y}_i = \sigma(w^T x_i + b) \) 是模型预测的概率。  


### 2. 优化方法：梯度下降  
通过梯度下降最小化损失函数，更新权重 \( w \) 和偏置 \( b \)，直到收敛。  


## **四、实现步骤（Python代码示例）**  
以下用乳腺癌数据集（二分类：良性/恶性）演示逻辑回归的完整流程。  


### **1. 环境准备与数据加载**  import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 加载数据（30个特征，1个二分类标签）
data = load_breast_cancer()
X = data.data  # 特征
y = data.target  # 标签（0: 恶性，1: 良性）

# 划分训练集和测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### **2. 数据预处理：标准化**  
逻辑回归对特征尺度敏感，需用 `StandardScaler` 标准化：  scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 用训练集拟合标准化器
X_test_scaled = scaler.transform(X_test)  # 测试集用相同参数转换

### **3. 模型训练**  # 初始化逻辑回归模型（L2正则化，C=1.0控制复杂度）
model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)

# 训练模型
model.fit(X_train_scaled, y_train)

### **4. 预测与评估**  # 预测测试集标签和概率
y_pred = model.predict(X_test_scaled)  # 分类标签（0/1）
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]  # 正类概率（良性）

# 评估准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")  # 输出示例：0.9737

# 混淆矩阵（真实标签 vs 预测标签）
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)

## **五、关键参数与调优**  
### 1. 正则化（`penalty`与`C`）  
- `penalty='l2'`（默认）：L2正则化，防止过拟合（权重平方和）。  
- `penalty='l1'`：L1正则化，可稀疏化权重（部分特征权重为0）。  
- `C`：正则化强度的倒数（`C`越小，正则化越强）。  


### 2. 最大迭代次数（`max_iter`）  
若模型未收敛（损失未稳定），需增大`max_iter`（如`max_iter=2000`）。  


## **六、优缺点与适用场景**  
### 优点  
- 简单高效，适合大规模数据。  
- 输出概率值，可解释性强（权重反映特征重要性）。  


### 缺点  
- 仅适用于线性可分问题（对非线性关系建模能力弱）。  
- 对特征尺度敏感（需标准化）。  


### 适用场景  
- 二分类问题（如垃圾邮件识别、疾病诊断）。  
- 需要概率输出的场景（如风控中的违约概率）。  


## **七、扩展：多分类与高阶特征**  
### 1. 多分类逻辑回归  
通过`multi_class='multinomial'`（多项逻辑回归）或`'ovr'`（一对多）扩展至多分类。  


### 2. 二阶逻辑回归  
通过`PolynomialFeatures`生成二阶多项式特征（如 \( x_1^2, x_1x_2 \)），捕捉非线性关系：  from sklearn.preprocessing import PolynomialFeatures

# 生成二阶多项式特征（包括交叉项和二次项）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)  # 原始特征扩展为二阶

## **总结**  
逻辑回归是经典的二分类算法，核心是通过Sigmoid函数将线性回归输出映射到概率区间。实际应用中需注意：  
- 标准化特征以避免尺度影响。  
- 通过正则化防止过拟合。  
- 输出概率值可灵活调整分类阈值（如医疗场景中降低漏诊率）。  


**附：完整代码**  # 完整代码（可直接运行）
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 加载数据
data = load_breast_cancer()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练逻辑回归模型
model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
model.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))    