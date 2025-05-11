#小題 3：訓練模型 (20 分)
#1. 選擇損失函數 ( MSELoss , CrossEntropyLoss 等) (2分)
#2. 選擇優化器（ SGD , Adam 等），並設置學習率。 (3分)
#3. 模型訓練至少 5 個 epoch，並在每個 epoch 結束時分別打印訓練集和驗證集的損失函數和準確率。 (15分)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器，学习率设为 0.001

for epoch in range(5):
    # 训练循环
    model.train()  # 开启训练模式（如启用 Dropout、BatchNorm 的训练模式）
    train_loss_total = 0.0
    train_correct = 0
    total_train = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # 数据移到指定设备（CPU/GPU）
        optimizer.zero_grad()  # 梯度清零
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_loss_total += loss.item()  # 累加训练损失
        _, predicted = outputs.max(1)  # 获取预测类别（概率最大的类别）
        total_train += labels.size(0)  # 累加总样本数
        train_correct += predicted.eq(labels).sum().item()  # 累加正确预测数

    train_loss = train_loss_total / (batch_idx + 1)  # 计算平均训练损失
    train_acc = train_correct / total_train  # 计算训练准确率

    # 打印训练信息（每 50 个 batch 打印一次，可根据实际调整）
    if batch_idx % 50 == 0:
        print(f"Epoch {epoch+1}, Batch {batch_idx}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%")

    # 验证循环
    model.eval()  # 开启评估模式（关闭 Dropout、BatchNorm 等的训练行为）
    val_loss_total = 0.0
    val_correct = 0
    total_val = 0
    with torch.no_grad():  # 禁用梯度计算（节省内存，加速计算）
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item()
            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_loss_total / len(val_loader)  # 计算平均验证损失
    val_acc = val_correct / total_val  # 计算验证准确率

    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%")
    
#格式正确性说明：
#缩进规则：
#每个 for 循环、if 语句内的代码块均缩进 4 个空格（如 model.train() 缩进在 for epoch 循环内，optimizer.zero_grad() 缩进在 for batch_idx 循环内）。
#确保同一层级的代码缩进一致（如训练循环和验证循环的代码块缩进对齐）。
#PyTorch API 规范：
#model.train() 和 model.eval() 正确切换模型模式（训练模式 / 评估模式）。
#optimizer.zero_grad()（梯度清零）、loss.backward()（反向传播）、optimizer.step()（参数更新）按标准流程编写。
#with torch.no_grad() 正确包裹验证循环（避免计算验证阶段的梯度）。
#变量计算与打印：
#训练损失 train_loss 和准确率 train_acc、验证损失 val_loss 和准确率 val_acc 的计算逻辑清晰。
#打印语句格式统一（使用 f-string，控制小数位数），便于观察训练过程。
