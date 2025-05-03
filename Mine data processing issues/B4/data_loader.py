import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(x_file, y_file):
    """
    加载并预处理X和Y数据
    
    Args:
        x_file: X数据文件路径
        y_file: Y数据文件路径
    
    Returns:
        处理后的X和Y数据
    """
    # 加载数据
    try:
        X_data = pd.read_excel(x_file, header=None)
        y_data = pd.read_excel(y_file, header=None)
        
        # 确保y数据只取第一列
        y_data = y_data.iloc[:, 0].values.reshape(-1, 1)
        
        # 标准化X数据
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_data)
        
        # 标准化Y数据
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_data)
        
        # 保存标准化器供后续使用
        global X_SCALER, Y_SCALER
        X_SCALER = scaler_X
        Y_SCALER = scaler_y
        
        return X_scaled, y_scaled
    
    except Exception as e:
        print(f"数据加载失败: {e}")
        raise

def inverse_transform_y(y_scaled):
    """将标准化后的y值转换回原始尺度"""
    return Y_SCALER.inverse_transform(y_scaled)
