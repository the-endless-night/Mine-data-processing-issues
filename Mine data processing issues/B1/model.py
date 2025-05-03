import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeatureTokenizer(nn.Module):
    """
    将表格数据中的连续特征转换为token embeddings
    """
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(1, embedding_dim)
        
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        # 将每个特征单独转换为embedding
        # 先扩展维度，将每个特征分开处理
        x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        embeddings = self.linear(x)  # [batch_size, input_dim, embedding_dim]
        
        return embeddings

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim必须能被num_heads整除"
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 线性变换
        Q = self.query(query)  # [batch_size, seq_len, embedding_dim]
        K = self.key(key)      # [batch_size, seq_len, embedding_dim]
        V = self.value(value)  # [batch_size, seq_len, embedding_dim]
        
        # 分割为多头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 缩放点积注意力
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.embedding_dim)
        x = self.fc(x)
        
        return x, attention

class PositionwiseFeedforward(nn.Module):
    """
    Position-wise Feedforward Network
    """
    def __init__(self, embedding_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    """
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.ff = PositionwiseFeedforward(embedding_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, mask=None):
        # 自注意力机制
        _src, _ = self.self_attn(src, src, src, mask)
        src = src + self.dropout(_src)
        src = self.norm1(src)
        
        # 前馈神经网络
        _src = self.ff(src)
        src = src + self.dropout(_src)
        src = self.norm2(src)
        
        return src

class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer
    """
    def __init__(self, input_dim, output_dim, embedding_dim=64, num_heads=8, 
                 num_layers=3, ff_dim=256, dropout=0.1):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(input_dim, embedding_dim)
        
        # 位置编码
        self.register_buffer('position_embedding', 
                            self._get_positional_encoding(input_dim, embedding_dim))
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, output_dim)
        )
        
    def _get_positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
        
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        batch_size = x.shape[0]
        
        # 特征标记化
        embeddings = self.feature_tokenizer(x)  # [batch_size, input_dim, embedding_dim]
        
        # 添加位置编码
        embeddings = embeddings + self.position_embedding.unsqueeze(0)
        
        # 通过Transformer编码器层
        for layer in self.encoder_layers:
            embeddings = layer(embeddings)
        
        # 平展并通过输出层
        embeddings = embeddings.reshape(batch_size, -1)
        output = self.fc(embeddings)
        
        return output

def create_model(device, input_dim=99, output_dim=1):
    """
    创建FTTransformer模型
    """
    model = FTTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        embedding_dim=64,
        num_heads=8,
        num_layers=4,
        ff_dim=256,
        dropout=0.1
    )
    model = model.to(device)
    return model
