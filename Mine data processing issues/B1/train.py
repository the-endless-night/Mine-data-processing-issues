import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class HuberLoss(nn.Module):
    """
    Huber损失函数：对离群点不敏感的损失函数
    对于小误差使用MSE，大误差使用MAE
    """
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        abs_error = torch.abs(y_pred - y_true)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        return loss.mean()

def train_model(model, train_loader, test_loader, device, num_epochs=50, robust_loss=True):
    """
    增强版训练模型：
    1. 使用稳健损失函数
    2. 添加早停机制
    3. 自适应学习率调整
    """
    if robust_loss:
        criterion = HuberLoss(delta=1.0)
        print("使用Huber损失函数进行鲁棒训练，减少离群点影响")
    else:
        criterion = nn.MSELoss()
        print("使用MSE损失函数进行训练")
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    # 早停机制
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        running_loss = 0.0
        
        # 使用tqdm显示进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'loss': loss.item()})
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # 更新学习率
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # 早停检查
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "d:/amath/B/B1/results/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停: {patience}个epoch内验证损失未改善")
                # 加载最佳模型
                model.load_state_dict(torch.load("d:/amath/B/B1/results/best_model.pth"))
                break
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.savefig('d:/amath/B/B1/results/损失曲线.png')
    plt.close()
    
    return model, train_losses, val_losses
