import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

class MineDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        矿山监测数据集加载器
        
        Args:
            data_path: 数据文件路径
            transform: 数据转换函数
        """
        # 加载数据，假设数据为CSV格式
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            self.data = pd.read_excel(data_path)
        else:
            raise ValueError("不支持的数据格式，请提供CSV或Excel文件")
            
        # 数据预处理
        self.data = self.data.select_dtypes(include=['float64', 'int64'])  # 仅保留数值列
        self.transform = transform
        
        # 标准化数据
        self.scaler = StandardScaler()
        self.normalized_data = self.scaler.fit_transform(self.data.values)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.normalized_data[idx, :]
        
        if self.transform:
            sample = self.transform(sample)
            
        return torch.FloatTensor(sample)
    
    def get_feature_dim(self):
        """返回数据的特征维度"""
        return self.data.shape[1]
    
    def inverse_transform(self, normalized_data):
        """将标准化后的数据转换回原始尺度"""
        return self.scaler.inverse_transform(normalized_data)
    
    def get_original_data(self):
        """获取原始数据"""
        return self.data.values

def get_data_loaders(data_path, batch_size=64, test_split=0.2, random_state=42):
    """
    创建训练和测试数据加载器
    
    Args:
        data_path: 数据文件路径
        batch_size: 批次大小
        test_split: 测试集比例
        random_state: 随机种子
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        dataset: 完整数据集
    """
    dataset = MineDataset(data_path)
    
    # 划分训练集和测试集
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[split:], indices[:split]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2
    )
    
    return train_loader, test_loader, dataset
