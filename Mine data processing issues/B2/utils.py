import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import time
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    """
    训练自动编码器模型并返回训练历史
    
    Args:
        model: 自动编码器模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        num_epochs: 训练轮数
        patience: 早停耐心值
        
    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    model = model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        
        for data in train_loader:
            # 将数据移到GPU
            data = data.to(device)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, data)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
        
        train_loss = train_loss / len(train_loader.sampler)
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data)
                val_loss += loss.item() * data.size(0)
        
        val_loss = val_loss / len(test_loader.sampler)
        
        # 记录损失
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停在 epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    total_time = time.time() - start_time
    print(f"训练完成，总时间: {total_time:.2f} 秒")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return model, history

def evaluate_model(model, data_loader, dataset, device):
    """
    评估模型并返回指标
    
    Args:
        model: 自动编码器模型
        data_loader: 数据加载器
        dataset: 数据集对象
        device: 设备
        
    Returns:
        metrics: 包含各种评估指标的字典
    """
    model.eval()
    original_data = []
    reconstructed_data = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            outputs = model(data)
            
            # 收集批次数据
            original_data.append(data.cpu().numpy())
            reconstructed_data.append(outputs.cpu().numpy())
    
    # 将所有批次连接起来
    original_data = np.concatenate(original_data, axis=0)
    reconstructed_data = np.concatenate(reconstructed_data, axis=0)
    
    # 将数据转换回原始尺度
    original_data_orig_scale = dataset.inverse_transform(original_data)
    reconstructed_data_orig_scale = dataset.inverse_transform(reconstructed_data)
    
    # 计算MSE和MAE
    mse = mean_squared_error(original_data, reconstructed_data)
    mae = mean_absolute_error(original_data, reconstructed_data)
    mse_orig_scale = mean_squared_error(original_data_orig_scale, reconstructed_data_orig_scale)
    mae_orig_scale = mean_absolute_error(original_data_orig_scale, reconstructed_data_orig_scale)
    
    # 计算压缩效率
    compression_ratio = model.get_compression_ratio()
    storage_saving_rate = model.get_storage_saving_rate()
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'mse_orig_scale': mse_orig_scale,
        'mae_orig_scale': mae_orig_scale,
        'compression_ratio': compression_ratio,
        'storage_saving_rate': storage_saving_rate,
        'original_data': original_data,
        'reconstructed_data': reconstructed_data,
        'original_data_orig_scale': original_data_orig_scale,
        'reconstructed_data_orig_scale': reconstructed_data_orig_scale
    }
    
    return metrics

def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练历史')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_data_comparison(original_data, reconstructed_data, feature_indices=None, num_samples=5, save_path=None):
    """
    绘制原始数据和重构数据的比较
    
    Args:
        original_data: 原始数据
        reconstructed_data: 重构数据
        feature_indices: 要绘制的特征索引列表
        num_samples: 要显示的样本数
        save_path: 保存图像的路径
    """
    if feature_indices is None:
        feature_indices = range(min(5, original_data.shape[1]))
    
    num_features = len(feature_indices)
    num_samples = min(num_samples, original_data.shape[0])
    
    plt.figure(figsize=(15, num_features * 3))
    
    for i, feature_idx in enumerate(feature_indices):
        plt.subplot(num_features, 1, i + 1)
        
        # 绘制前num_samples个样本
        x = range(num_samples)
        plt.plot(x, original_data[:num_samples, feature_idx], 'b-', label='原始数据', alpha=0.7)
        plt.plot(x, reconstructed_data[:num_samples, feature_idx], 'r--', label='重构数据', alpha=0.7)
        
        plt.title(f'特征 {feature_idx} 的原始数据和重构数据对比')
        plt.xlabel('样本索引')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def analyze_compression_efficiency(model, metrics):
    """分析压缩效率"""
    compression_ratio = metrics['compression_ratio']
    storage_saving_rate = metrics['storage_saving_rate']
    mse = metrics['mse']
    
    print(f"压缩分析结果:")
    print(f"  - 输入维度: {model.input_dim}")
    print(f"  - 编码维度: {model.encoding_dim}")
    print(f"  - 压缩比: {compression_ratio:.2f}:1")
    print(f"  - 存储空间节省率: {storage_saving_rate:.2%}")
    print(f"  - 重构MSE: {mse:.6f}")
    
    return {
        'compression_ratio': compression_ratio,
        'storage_saving_rate': storage_saving_rate,
        'mse': mse
    }
