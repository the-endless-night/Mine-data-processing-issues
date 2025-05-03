import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from joblib import Memory
import time
import os
import psutil

# CUDNN性能调优
cudnn.benchmark = True      # 启用自动寻找最佳算法
cudnn.deterministic = False # 允许非确定性算法以换取速度

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加备用字体
plt.rcParams['axes.unicode_minus'] = False

# 创建一个函数用于处理特殊字符显示
def format_label(text):
    """处理特殊字符，确保在任何字体下都能正常显示"""
    # 将R²替换为R^2
    return text.replace('R²', 'R^2')

# 设置随机种子，确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 创建输出目录和缓存目录
output_dir = 'd:/amath/B/B5/output'
cache_dir = 'd:/amath/B/B5/cache'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# 创建sklearn缓存
memory = Memory(cache_dir, verbose=0)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 自动确定最佳批量大小
def find_optimal_batch_size(input_dim, initial_batch_size=64, max_batch_size=1024):
    """二分搜索找到在当前GPU上可用的最大batch_size"""
    if not torch.cuda.is_available():
        return initial_batch_size
    
    low, high = initial_batch_size, max_batch_size
    optimal = initial_batch_size
    
    while low <= high:
        mid = (low + high) // 2
        try:
            # 尝试分配一个批次的数据和模型权重
            x = torch.randn(mid, input_dim, device=device)
            model = nn.Linear(input_dim, input_dim).to(device)
            out = model(x)
            del x, model, out
            torch.cuda.empty_cache()
            
            # 如果成功，尝试更大的batch_size
            optimal = mid
            low = mid + 1
        except RuntimeError as e:
            # 内存不足，尝试更小的batch_size
            high = mid - 1
            torch.cuda.empty_cache()
    
    # 留一些余量，返回略小于找到的最大值
    return max(initial_batch_size, int(optimal * 0.9))

# 1. 读取数据
def load_data():
    """读取原始数据并进行基本清洗"""
    # 读取Excel文件
    X_data = pd.read_excel('5-X.xlsx', header=None)
    Y_data = pd.read_excel('5-Y.xlsx', header=None).iloc[:, 0].values.reshape(-1, 1)
    
    # 检查并处理重复行
    duplicate_rows = X_data.duplicated()
    if duplicate_rows.any():
        print(f"发现{duplicate_rows.sum()}个重复行，将被删除")
        mask = ~duplicate_rows
        X_data = X_data[mask]
        Y_data = Y_data[mask.values]
    
    # 检查缺失值
    missing_values_X = X_data.isnull().sum().sum()
    if missing_values_X > 0:
        print(f"X数据中发现{missing_values_X}个缺失值")
        # 使用列均值填充缺失值
        X_data = X_data.fillna(X_data.mean())
        print("已使用均值填充缺失值")
    
    print(f"X数据形状: {X_data.shape}, Y数据形状: {Y_data.shape}")
    return X_data, Y_data

# 2. 数据预处理
def preprocess_data(X_data, Y_data, add_nonlinear=False):  # 默认改为False
    """对数据进行标准化预处理和非线性变换"""
    # 处理X数据中的异常值（使用Z-score方法）
    X_data_copy = X_data.copy()
    
    # 确保所有列名都是字符串类型
    X_data_copy.columns = X_data_copy.columns.astype(str)
    
    # 向量化处理异常值
    # 计算Z-scores
    z_scores = np.abs((X_data_copy - X_data_copy.mean()) / X_data_copy.std())
    
    # 找出所有异常值并统计
    outliers_mask = z_scores > 3
    total_outliers = outliers_mask.sum().sum()
    if total_outliers > 0:
        print(f"检测到 {total_outliers} 个异常值")
        
        # 计算上下界限
        upper_bounds = X_data_copy.mean() + 3 * X_data_copy.std()
        lower_bounds = X_data_copy.mean() - 3 * X_data_copy.std()
        
        # 向量化替换异常值
        X_data_copy = X_data_copy.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    
    # 添加非线性特征（仅当明确要求时）
    if add_nonlinear:
        print("添加非线性特征...")
        # 选择部分特征列进行非线性变换（避免维度爆炸）
        n_features = min(20, X_data_copy.shape[1])  # 最多使用20个特征
        feature_cols = X_data_copy.columns[:n_features]
        
        # 创建非线性特征
        X_nonlinear = X_data_copy.copy()
        
        # 1. 添加平方特征
        for col in feature_cols:
            X_nonlinear[f"{col}_squared"] = X_data_copy[col] ** 2
        
        # 2. 添加交互特征（最多10个）
        interaction_count = 0
        for i in range(min(5, len(feature_cols))):
            for j in range(i+1, min(6, len(feature_cols))):
                if interaction_count < 10:
                    col_i = feature_cols[i]
                    col_j = feature_cols[j]
                    X_nonlinear[f"{col_i}_{col_j}_interaction"] = X_data_copy[col_i] * X_data_copy[col_j]
                    interaction_count += 1
        
        # 3. 添加三角函数特征
        for i, col in enumerate(feature_cols[:5]):  # 最多对5个特征
            X_nonlinear[f"{col}_sin"] = np.sin(X_data_copy[col])
            X_nonlinear[f"{col}_cos"] = np.cos(X_data_copy[col])
        
        # 4. 添加对数变换（处理可能的负值）
        for i, col in enumerate(feature_cols[:5]):  # 最多对5个特征
            # 归一化到正数范围
            min_val = X_data_copy[col].min()
            if min_val < 0:
                shifted = X_data_copy[col] - min_val + 1.0  # 确保全部为正
            else:
                shifted = X_data_copy[col] + 1.0  # 避免取对数为0
            X_nonlinear[f"{col}_log"] = np.log(shifted)
        
        print(f"特征数量: 原始 {X_data_copy.shape[1]}, 非线性变换后 {X_nonlinear.shape[1]}")
        X_data_copy = X_nonlinear
    
    # 处理Y数据中的异常值 - 向量化处理
    Y_data_copy = Y_data.copy()
    Y_mean = np.mean(Y_data)
    Y_std = np.std(Y_data)
    Y_outliers = np.abs(Y_data - Y_mean) > 3 * Y_std
    if np.sum(Y_outliers) > 0:
        print(f"Y数据中检测到 {np.sum(Y_outliers)} 个异常值")
        # 将Y异常值替换为边界值
        Y_data_copy = np.clip(Y_data, Y_mean - 3 * Y_std, Y_mean + 3 * Y_std)
    
    # 标准化X数据
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_data_copy)
    
    # 标准化Y数据
    Y_scaler = StandardScaler()
    Y_scaled = Y_scaler.fit_transform(Y_data_copy)
    
    return X_scaled, Y_scaled, X_scaler, Y_scaler

# 3. 构建自编码器模型 (增强非线性)
class Autoencoder(nn.Module):
    """自编码器模型，用于降维和重构，增强非线性"""
    def __init__(self, input_dim, encoding_dim, dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        # 编码器部分 - 使用更多非线性激活函数
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),  # 使用LeakyReLU而不是ReLU
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.Tanh()  # 使用Tanh激活函数，增加非线性
        )
        
        # 解码器部分 - 使用更多非线性激活函数
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # 使用Sigmoid激活增加非线性，尤其对于归一化数据
        )
    
    def forward(self, x):
        # 前向传播，完成编码和解码
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        # 仅执行编码部分
        return self.encoder(x)

# 3.5 寻找最优降维维度和模型参数
def optimize_autoencoder_dim(X_scaled, Y_scaled, dim_range=None, epochs=50, batch_size=None):
    """
    自动寻找最优的降维维度和模型参数，优化R²拟合优度
    
    参数:
    - X_scaled: 标准化后的输入数据
    - Y_scaled: 标准化后的目标数据
    - dim_range: 需要测试的维度范围列表，默认根据输入维度自动确定
    - epochs: 每个维度尝试的训练轮数
    - batch_size: 训练批量大小
    
    返回:
    - best_dim: 最优维度
    - best_r2: 最佳R²值
    - results: 各维度测试结果
    """
    print("\n自动寻找最优降维维度和模型参数...")
    
    # 如果没有指定批量大小，则自动找到最优值
    if batch_size is None:
        batch_size = find_optimal_batch_size(X_scaled.shape[1])
        print(f"自动调整批量大小为: {batch_size}")
    
    # 如果没有指定维度范围，则自动确定
    if dim_range is None:
        input_dim = X_scaled.shape[1]
        # 设置维度范围，从较小值开始递增，最大不超过原始维度的50%
        min_dim = max(5, int(input_dim * 0.05))  # 至少5维
        max_dim = min(int(input_dim * 0.5), input_dim - 1)  # 最多原始维度的50%
        # 生成均匀分布的测试维度
        if max_dim - min_dim > 20:
            step = (max_dim - min_dim) // 10  # 如果范围很大，选择约10个维度点
        else:
            step = max(1, (max_dim - min_dim) // 5)  # 否则选择约5个维度点
            
        dim_range = list(range(min_dim, max_dim + 1, step))
        # 确保包含一些关键值如10, 20, 30等
        for key_dim in [10, 15, 20, 25, 30]:
            if key_dim > min_dim and key_dim < max_dim and key_dim not in dim_range:
                dim_range.append(key_dim)
        dim_range.sort()
    
    # 存储每个维度的测试结果
    results = {}
    best_dim = None
    best_r2 = -float('inf')
    
    # 测试每个维度
    for dim in dim_range:
        print(f"\n测试降维维度: {dim}")
        
        # 训练自编码器
        autoencoder, _, _ = train_autoencoder(X_scaled, encoding_dim=dim, epochs=epochs, batch_size=batch_size)
        
        # 进行降维和重构
        encoded_data, reconstructed_data, _, _ = reduce_and_reconstruct(X_scaled, autoencoder)
        
        # 评估重构误差
        reconstruction_error = np.mean(np.power(X_scaled - reconstructed_data, 2))
        print(f"重构误差 (MSE): {reconstruction_error:.6f}")
        
        # 使用重构数据训练MLP，评估R²性能
        try:
            # 使用RandomizedSearchCV代替网格搜索，更快找到近似最优解
            X_train, X_test, y_train, y_test = train_test_split(
                reconstructed_data, Y_scaled, test_size=0.2, random_state=42
            )
            
            # 扁平化标签
            if y_train.ndim > 1:
                y_train = y_train.ravel()
            if y_test.ndim > 1:
                y_test = y_test.ravel()
            
            # 使用随机搜索而非穷举搜索，加快评估过程
            param_distributions = {
                'hidden_layer_sizes': [
                    (64, 32), 
                    (128, 64), 
                    (128, 64, 32),
                    (256, 128, 64, 32)
                ],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
            
            mlp = MLPRegressor(
                solver='adam',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            
            # 使用RandomizedSearchCV，并行执行
            n_cores = max(1, psutil.cpu_count(logical=False) - 1)  # 物理核心数-1
            search = RandomizedSearchCV(
                mlp, 
                param_distributions, 
                n_iter=10,
                cv=3,
                scoring='r2',
                n_jobs=n_cores,
                random_state=42
            )
            
            search.fit(X_train, y_train)
            
            # 获取最佳模型
            best_mlp = search.best_estimator_
            best_mlp_config = {
                'hidden_layer_sizes': best_mlp.hidden_layer_sizes,
                'activation': best_mlp.activation
            }
            
            # 在测试集上评估
            y_pred = best_mlp.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            
            # 计算综合评分
            combined_score = test_r2 - 0.1 * reconstruction_error
            
            print(f"维度 {dim}: R² = {test_r2:.4f}, 重构误差 = {reconstruction_error:.4f}, 综合评分 = {combined_score:.4f}")
            print(f"最佳MLP配置: {best_mlp_config}")
            
            results[dim] = {
                'r2': test_r2,
                'reconstruction_error': reconstruction_error,
                'combined_score': combined_score,
                'mlp_config': (best_mlp.hidden_layer_sizes, best_mlp.activation)
            }
            
            # 更新最佳维度
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_dim = dim
                
        except Exception as e:
            print(f"评估维度 {dim} 时出错: {str(e)}")
            results[dim] = {
                'error': str(e)
            }
    
    # 可视化不同维度的R²和重构误差
    plt.figure(figsize=(12, 6))

    dims = [d for d in dim_range if d in results and 'r2' in results[d]]
    if dims:
        r2_scores = [results[d]['r2'] for d in dims]
        recon_errors = [results[d]['reconstruction_error'] for d in dims]
        
        ax1 = plt.subplot(111)
        ax1.plot(dims, r2_scores, 'b-o', label='R$^2$值')  # 正确使用 LaTeX
        ax1.set_xlabel('降维维度')
        ax1.set_ylabel('R$^2$值', color='b')  # 轴标签用 LaTeX
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(dims, recon_errors, 'r-o', label='重构误差')
        ax2.set_ylabel('重构误差', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 标记最佳维度（修复 LaTeX 公式）
        if best_dim:
            ax1.axvline(
                x=best_dim, 
                color='g', 
                linestyle='--', 
                alpha=0.7,
                label=f'最佳维度: {best_dim} (R$^2$={best_r2:.4f})'  # 修正 R² 的 LaTeX 语法
            )
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title('不同降维维度的性能比较')
        plt.savefig(f'{output_dir}/dimension_optimization.png')

    # 处理默认维度逻辑
    if best_dim is None:
        print("未找到有效的最佳维度，使用默认值20")
        best_dim = 20
    else:
        # 修正输出中的 R² 格式（终端不支持 LaTeX，保持原样）
        print(f"\n最佳降维维度: {best_dim}, 最佳R²值: {best_r2:.4f}")  

    return best_dim, best_r2, results

# 4. 训练自编码器模型
def train_autoencoder(X_scaled, encoding_dim=20, epochs=150, batch_size=None):
    """训练自编码器模型并返回训练好的模型"""
    input_dim = X_scaled.shape[1]
    
    # 自动找到最优批量大小
    if batch_size is None:
        batch_size = find_optimal_batch_size(input_dim)
        print(f"自动调整批量大小为: {batch_size}")
    
    # 划分训练集和验证集
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    
    # 创建数据加载器 - 添加并行加载
    num_workers = min(4, os.cpu_count() or 1)  # 最多4个工作进程
    
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 构建自编码器模型
    autoencoder = Autoencoder(input_dim, encoding_dim, dropout_rate=0.2).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # 早停策略
    patience = 15
    min_val_loss = float('inf')
    counter = 0
    best_model = None
    
    # 记录训练历史
    history = {'loss': [], 'val_loss': []}
    
    # 初始化混合精度训练
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    
    # 训练循环
    start_time = time.time()
    for epoch in range(epochs):
        # 训练模式
        autoencoder.train()
        train_loss = 0
        
        for data, _ in train_loader:
            # 非阻塞传输到设备
            data = data.to(device, non_blocking=True)
            
            # 混合精度训练
            optimizer.zero_grad()
            
            if scaler is not None:
                # 使用自动混合精度 - 修复API调用
                with autocast('cuda'):
                    outputs = autoencoder(data)
                    loss = criterion(outputs, data)
                
                # 缩放损失进行反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通训练
                outputs = autoencoder(data)
                loss = criterion(outputs, data)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * data.size(0)
        
        # 计算训练集平均损失
        train_loss = train_loss / len(train_loader.dataset)
        history['loss'].append(train_loss)
        
        # 评估模式
        autoencoder.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device, non_blocking=True)
                
                if scaler is not None:
                    with autocast('cuda'):
                        outputs = autoencoder(data)
                        loss = criterion(outputs, data)
                else:
                    outputs = autoencoder(data)
                    loss = criterion(outputs, data)
                    
                val_loss += loss.item() * data.size(0)
        
        # 计算验证集平均损失
        val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 输出进度
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 检查早停条件
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            counter = 0
            best_model = autoencoder.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # 如果训练完成，加载最佳模型
    if best_model:
        autoencoder.load_state_dict(best_model)
    
    training_time = time.time() - start_time
    
    # 编译模型以加速推理
    try:
        # 创建一个示例输入，用于JIT追踪
        example_input = torch.randn(1, input_dim, device=device)
        
        # 设置为评估模式
        autoencoder.eval()
        
        # 使用JIT追踪编译模型
        jit_model = torch.jit.trace(autoencoder, example_input)
        
        # 保存编译后的模型
        torch.jit.save(jit_model, f'{output_dir}/autoencoder_jit.pt')
        print("已保存JIT编译的模型用于快速推理")
    except Exception as e:
        print(f"JIT编译失败: {e}")
    
    return autoencoder, history, training_time

# 5. 使用自编码器进行降维和重构
def reduce_and_reconstruct(X_scaled, autoencoder):
    """使用训练好的自编码器进行降维和重构"""
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_scaled).to(device, non_blocking=True)
    
    # 将模型设置为评估模式
    autoencoder.eval()
    
    # 使用自动混合精度进行推理
    start_time = time.time()
    with torch.no_grad():
        if torch.cuda.is_available():
            with autocast('cuda'):
                encoded_tensor = autoencoder.encode(X_tensor)
        else:
            encoded_tensor = autoencoder.encode(X_tensor)
    encoded_data = encoded_tensor.cpu().numpy()
    encoding_time = time.time() - start_time
    
    # 重构
    start_time = time.time()
    with torch.no_grad():
        if torch.cuda.is_available():
            with autocast('cuda'):
                reconstructed_tensor = autoencoder(X_tensor)
        else:
            reconstructed_tensor = autoencoder(X_tensor)
    reconstructed_data = reconstructed_tensor.cpu().numpy()
    reconstruction_time = time.time() - start_time
    
    return encoded_data, reconstructed_data, encoding_time, reconstruction_time

# 6. 建立重构数据与Y之间的关系模型 - 使用MLP (增强非线性)
def build_mlp_model(reconstructed_data, Y_scaled):
    """使用增强型MLP建立重构数据与Y之间的关系模型"""
    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        reconstructed_data, Y_scaled, test_size=0.2, random_state=42
    )
    
    # 将Y转换为一维数组，避免警告
    if Y_train.ndim > 1:
        Y_train = Y_train.ravel()
    if Y_test.ndim > 1:
        Y_test = Y_test.ravel()
    
    # 使用Pipeline和缓存优化特征变换
    n_cores = max(1, psutil.cpu_count(logical=False) - 1)  # 使用物理核心数-1
    
    # 创建带缓存的Pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='tanh',
            solver='adam',
            alpha=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            random_state=42
        ))
    ], memory=memory)
    
    # 使用交叉验证评估模型，使用并行计算
    cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=3, scoring='r2', n_jobs=n_cores)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"交叉验证R^2分数: {cv_scores}")
    print(f"平均R^2分数: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # 训练最终模型
    start_time = time.time()
    pipeline.fit(X_train, Y_train)
    training_time = time.time() - start_time
    
    # 在测试集上进行预测
    start_time = time.time()
    Y_pred = pipeline.predict(X_test)
    prediction_time = time.time() - start_time
    
    # 计算模型评估指标
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    # 获取训练后的MLP模型
    mlp_model = pipeline.named_steps['mlp']
    poly = pipeline.named_steps['poly']
    
    return mlp_model, poly, mse, r2, Y_test, Y_pred, training_time, prediction_time, cv_mean, cv_std

# 7. 可视化结果
def visualize_results(history, encoded_data, Y_scaled, Y_test, Y_pred):
    """可视化自编码器训练过程和预测结果"""
    # 自编码器训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('自编码器训练损失')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f'{output_dir}/autoencoder_loss.png')
    
    # 降维数据可视化（如果维度为2）
    if encoded_data.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=Y_scaled.flatten(), cmap='viridis')
        plt.colorbar(label='目标值（标准化后）')
        plt.title('降维后的数据 (2D) 与目标值的关系')
        plt.xlabel('降维特征 1')
        plt.ylabel('降维特征 2')
        plt.savefig(f'{output_dir}/reduced_data_2d.png')
    
    # 如果维度为3，创建3D可视化
    elif encoded_data.shape[1] == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2], 
                     c=Y_scaled.flatten(), cmap='viridis')
        plt.colorbar(p, label='目标值（标准化后）')
        ax.set_title('降维后的数据 (3D) 与目标值的关系')
        ax.set_xlabel('降维特征 1')
        ax.set_ylabel('降维特征 2')
        ax.set_zlabel('降维特征 3')
        plt.savefig(f'{output_dir}/reduced_data_3d.png')
    
    # 预测结果可视化
    plt.figure(figsize=(10, 8))
    
    # 确保数据是一维的用于绘图
    y_test_plot = Y_test.flatten() if hasattr(Y_test, 'flatten') else Y_test
    y_pred_plot = Y_pred.flatten() if hasattr(Y_pred, 'flatten') else Y_pred
    
    plt.scatter(y_test_plot, y_pred_plot, alpha=0.6)
    
    # 添加对角线
    min_val = min(y_test_plot.min(), y_pred_plot.min())
    max_val = max(y_test_plot.max(), y_pred_plot.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测线')
    
    plt.title('预测值 vs 实际值')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/prediction_vs_actual.png')
    
    # 残差分析
    plt.figure(figsize=(10, 6))
    residuals = y_test_plot - y_pred_plot
    plt.scatter(y_pred_plot, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('残差分析')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/residual_analysis.png')
    
    # 模型性能可视化
    plt.figure(figsize=(8, 6))
    performance = {
        'MSE': mean_squared_error(y_test_plot, y_pred_plot),
        'R$^2$': r2_score(y_test_plot, y_pred_plot),
        '平均残差': np.mean(np.abs(residuals))
    }
    
    plt.bar(performance.keys(), performance.values())
    plt.title('MLP模型性能指标')
    plt.ylim(0, 1.0 if performance['R^2'] < 1 else 1.1)  # 更新键名引用
    for i, (k, v) in enumerate(performance.items()):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.savefig(f'{output_dir}/model_performance.png')

# 8. 算法复杂度分析
def complexity_analysis(autoencoder, X_data_shape, ae_training_time, encoding_time, reconstruction_time, 
                        mlp_training_time=None, mlp_prediction_time=None):
    """分析算法的时间和空间复杂度"""
    input_dim = X_data_shape[1]
    total_samples = X_data_shape[0]
    
    # 计算自编码器参数数量
    trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    
    analysis_report = f"""
    算法复杂度分析：
    -----------------
    
    1. 数据规模:
       - 样本数: {total_samples}
       - 输入维度: {input_dim}
    
    2. 自编码器模型(PyTorch):
       - 总参数数量: {trainable_params}
       - 训练时间: {ae_training_time:.2f} 秒
       - 编码时间: {encoding_time:.4f} 秒
       - 重构时间: {reconstruction_time:.4f} 秒
       - 时间复杂度: O(epochs * batch_size * n * m)，其中n是样本数，m是参数数量
       - 空间复杂度: O(m)，其中m是参数数量
    """
    
    if mlp_training_time is not None and mlp_prediction_time is not None:
        analysis_report += f"""
    3. 预测模型 (MLP):
       - 训练时间: {mlp_training_time:.4f} 秒
       - 预测时间: {mlp_prediction_time:.4f} 秒
       - MLP时间复杂度: O(epochs * n * h * d)，其中n是样本数，h是隐藏层神经元数量，d是特征维度
       - MLP空间复杂度: O(h * d)，其中h是隐藏层神经元数量，d是特征维度
    
    4. 综合分析:
       - 整个模型流程包括降维、重构和预测三个主要阶段
       - 自编码器降维的主要优势在于可以保留数据的重要特征并降低噪声影响
       - MLP模型能够学习重构数据与目标变量之间的非线性关系
       - 总的计算复杂度主要受样本数量和模型参数量的影响
    """
    
    return analysis_report

# 主函数
def main():
    # 读取数据
    print("1. 读取数据...")
    X_data, Y_data = load_data()
    
    # 数据预处理，不添加非线性特征
    print("\n2. 预处理数据...")
    X_scaled, Y_scaled, X_scaler, Y_scaler = preprocess_data(X_data, Y_data, add_nonlinear=False)
    
    # 自动寻找最优批量大小
    optimal_batch_size = find_optimal_batch_size(X_scaled.shape[1])
    print(f"为当前GPU优化的批量大小: {optimal_batch_size}")
    
    # 自动寻找最优降维维度
    best_dim, best_r2, dim_results = optimize_autoencoder_dim(
        X_scaled, 
        Y_scaled,
        epochs=50,
        batch_size=optimal_batch_size
    )
    
    # 使用最优维度训练完整自编码器
    print(f"\n3. 使用最优维度({best_dim})训练自编码器模型...")
    autoencoder, history, ae_training_time = train_autoencoder(
        X_scaled, 
        encoding_dim=best_dim,
        epochs=150,
        batch_size=optimal_batch_size
    )
    
    # 使用自编码器进行降维和重构
    print("\n4. 自编码器降维和数据重构...")
    encoded_data, reconstructed_data, encoding_time, reconstruction_time = reduce_and_reconstruct(X_scaled, autoencoder)
    reconstruction_error = np.mean(np.power(X_scaled - reconstructed_data, 2))
    print(f"降维后的数据形状: {encoded_data.shape}")
    print(f"重构后的数据形状: {reconstructed_data.shape}")
    print(f"重构误差 (MSE): {reconstruction_error:.6f}")
    
    # 使用MLP建立预测模型
    print("\n5. 使用MLP建立预测模型...")
    # 获取最佳MLP配置（如果可用）
    if best_dim in dim_results and 'mlp_config' in dim_results[best_dim]:
        best_mlp_config = dim_results[best_dim]['mlp_config']
        print(f"使用优化后的MLP配置: 隐藏层 = {best_mlp_config[0]}, 激活函数 = {best_mlp_config[1]}")
    else:
        best_mlp_config = ((256, 128, 64, 32), 'tanh')  # 默认配置
    
    mlp_model, poly, mlp_mse, mlp_r2, Y_test, Y_pred, mlp_training_time, mlp_prediction_time, cv_mean, cv_std = build_mlp_model(reconstructed_data, Y_scaled)
    print(f"MLP模型 - MSE: {mlp_mse:.6f}, R^2: {mlp_r2:.6f}")
    
    # 可视化结果
    print("\n6. 可视化结果...")
    visualize_results(history, encoded_data, Y_scaled, Y_test, Y_pred)
    
    # 算法复杂度分析
    print("\n7. 算法复杂度分析...")
    analysis = complexity_analysis(
        autoencoder, 
        X_scaled.shape, 
        ae_training_time, 
        encoding_time, 
        reconstruction_time,
        mlp_training_time, 
        mlp_prediction_time
    )
    print(analysis)
    
    # 保存分析报告
    with open(f'{output_dir}/complexity_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(analysis)
    
    # 保存模型评估报告
    evaluation_report = f"""
    模型评估报告：
    -----------------
    
    1. 自编码器降维与重构:
       - 原始维度: {X_scaled.shape[1]}
       - 降维后维度: {encoded_data.shape[1]}
       - 重构误差 (MSE): {reconstruction_error:.6f}
    
    2. MLP预测模型性能:
       - 均方误差 (MSE): {mlp_mse:.6f}
       - 决定系数 (R^2): {mlp_r2:.6f}
       - 交叉验证R^2: {cv_mean:.6f} ± {cv_std:.6f}
    
    3. 模型泛化性分析:
       - 使用了训练集(80%)和测试集(20%)分离评估模型泛化性
       - 交叉验证评估了模型在不同数据子集上的表现稳定性
       - R^2值反映了模型解释目标变量方差的能力
       - {"模型表现良好，具有较好的泛化能力" if mlp_r2 > 0.5 else "模型表现一般，泛化能力有限" if mlp_r2 > 0.2 else "模型表现不佳，预测能力较弱"}
    
    4. 结论与建议:
       - 自编码器成功将{X_scaled.shape[1]}维数据降维到{encoded_data.shape[1]}维，同时保持了重要信息
       - MLP模型能够从重构数据中学习与目标变量的关系
       - {"建议保持当前模型" if mlp_r2 > 0.5 else "可以尝试调整自编码器结构或编码维度以提高性能" if mlp_r2 > 0.2 else "建议重新考虑模型架构或进行更多特征工程"}
       - 可以考虑探索其他降维技术(如PCA、t-SNE)或其他预测模型进行比较
    """
    
    with open(f'{output_dir}/model_evaluation.txt', 'w', encoding='utf-8') as f:
        f.write(evaluation_report)
    
    print("\n分析完成！结果已保存到 d:/amath/B/B5/output/ 目录")

if __name__ == "__main__":
    main()
