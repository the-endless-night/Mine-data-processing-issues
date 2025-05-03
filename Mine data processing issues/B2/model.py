import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dims=None):
        """
        矿山监测数据自动编码器
        
        Args:
            input_dim: 输入特征维度
            encoding_dim: 编码维度（压缩后的维度）
            hidden_dims: 编码器和解码器中隐藏层的维度列表
        """
        super(Autoencoder, self).__init__()
        
        # 如果没有指定hidden_dims，则创建默认的结构
        if hidden_dims is None:
            # 根据输入维度和编码维度自动创建一个合理的结构
            hidden_dims = [max(input_dim // 2, encoding_dim * 2)]
        
        # 构建编码器
        encoder_layers = []
        
        # 第一层：输入层到第一个隐藏层
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        encoder_layers.append(nn.ReLU())
        
        # 中间层
        for i in range(len(hidden_dims) - 1):
            encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            encoder_layers.append(nn.ReLU())
        
        # 最后一层：最后一个隐藏层到编码层
        encoder_layers.append(nn.Linear(hidden_dims[-1], encoding_dim))
        
        # 构建解码器（与编码器对称）
        decoder_layers = []
        
        # 第一层：编码层到第一个隐藏层
        decoder_layers.append(nn.Linear(encoding_dim, hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())
        
        # 中间层
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
            decoder_layers.append(nn.ReLU())
        
        # 最后一层：最后一个隐藏层到输出层
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        
        # 创建编码器和解码器模块
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
    def forward(self, x):
        """前向传播"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """仅编码数据"""
        return self.encoder(x)
    
    def decode(self, z):
        """仅解码数据"""
        return self.decoder(z)
    
    def get_compression_ratio(self):
        """计算压缩比"""
        return self.input_dim / self.encoding_dim
    
    def get_storage_saving_rate(self):
        """计算存储空间节省率"""
        return 1 - (self.encoding_dim / self.input_dim)
