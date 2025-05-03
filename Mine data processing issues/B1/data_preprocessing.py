import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib
import seaborn as sns
from scipy.signal import savgol_filter

# 设置支持中文显示的字体
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def load_data(a_path, b_path):
    """
    加载Excel数据文件 - 假设第0行开始就是数据，没有列名
    """
    # 读取Excel文件，不使用表头
    df_a = pd.read_excel(a_path, header=None)
    df_b = pd.read_excel(b_path, header=None)
    
    # 为列生成默认名称
    df_a.columns = [f'col_{i}' for i in range(df_a.shape[1])]
    df_b.columns = [f'target_{i}' for i in range(df_b.shape[1])]
    
    print(f"A数据形状: {df_a.shape}")
    print(f"B数据形状: {df_b.shape}")
    
    # 验证A和B的行数是否相同
    if len(df_a) != len(df_b):
        print(f"警告: A数据({len(df_a)}行)和B数据({len(df_b)}行)行数不一致!")
    else:
        print(f"数据行数一致: {len(df_a)}行")
    
    return df_a, df_b

def clean_data(df_a, df_b, handle_outliers=True, outlier_method='zscore', contamination=0.05):
    """
    增强版数据清洗:
    1. 处理缺失值
    2. 强化异常值检测与处理
    3. 平滑处理减少噪声
    4. 添加鲁棒性特征
    """
    print("原始数据统计信息:")
    print(f"A数据缺失值统计: {df_a.isnull().sum().sum()}")
    print(f"B数据缺失值统计: {df_b.isnull().sum().sum()}")
    
    # 确保A和B的行索引一致
    df_a = df_a.reset_index(drop=True)
    df_b = df_b.reset_index(drop=True)
    
    # 1. 处理缺失值 - 填充或删除
    df_a = df_a.fillna(df_a.mean())
    df_b = df_b.fillna(df_b.mean())
    
    # 2. 增强异常值检测方法
    outlier_indices = []
    
    if handle_outliers:
        if outlier_method == 'zscore':
            # Z-score方法检测异常值
            z_scores_a = np.abs(stats.zscore(df_a, nan_policy='omit'))
            z_scores_b = np.abs(stats.zscore(df_b, nan_policy='omit'))
            
            # 检测异常值 - 同时考虑A和B的异常
            outlier_mask_a = z_scores_a > 3
            outlier_mask_b = z_scores_b > 3
            
            # 整合异常值掩码 - 任一数据异常则标记
            outlier_mask = np.any(outlier_mask_a, axis=1) | np.any(outlier_mask_b, axis=1)
            outlier_indices = np.where(outlier_mask)[0]
            
        elif outlier_method == 'isolation_forest':
            # 使用IsolationForest检测多维异常
            clf = IsolationForest(contamination=contamination, random_state=42)
            X_combined = pd.concat([df_a, df_b], axis=1)
            preds = clf.fit_predict(X_combined)
            outlier_indices = np.where(preds == -1)[0]
            
        elif outlier_method == 'lof':
            # 使用LocalOutlierFactor检测局部异常
            clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            X_combined = pd.concat([df_a, df_b], axis=1)
            preds = clf.fit_predict(X_combined)
            outlier_indices = np.where(preds == -1)[0]
            
        elif outlier_method == 'robust':
            # 组合多种方法检测异常
            # 1. Z-score
            z_scores_a = np.abs(stats.zscore(df_a, nan_policy='omit'))
            z_scores_b = np.abs(stats.zscore(df_b, nan_policy='omit'))
            outlier_mask_z = (np.any(z_scores_a > 3, axis=1) | 
                              np.any(z_scores_b > 3, axis=1))
            
            # 2. IQR方法
            Q1_a = df_a.quantile(0.25)
            Q3_a = df_a.quantile(0.75)
            IQR_a = Q3_a - Q1_a
            outlier_mask_iqr_a = ((df_a < (Q1_a - 1.5 * IQR_a)) | 
                                  (df_a > (Q3_a + 1.5 * IQR_a))).any(axis=1)
            
            Q1_b = df_b.quantile(0.25)
            Q3_b = df_b.quantile(0.75)
            IQR_b = Q3_b - Q1_b
            outlier_mask_iqr_b = ((df_b < (Q1_b - 1.5 * IQR_b)) | 
                                  (df_b > (Q3_b + 1.5 * IQR_b))).any(axis=1)
            
            outlier_mask_iqr = outlier_mask_iqr_a | outlier_mask_iqr_b
            
            # 3. 隔离森林
            clf = IsolationForest(contamination=contamination, random_state=42)
            X_combined = pd.concat([df_a, df_b], axis=1)
            preds_if = clf.fit_predict(X_combined)
            outlier_mask_if = preds_if == -1
            
            # 组合异常检测结果 - 至少两种方法检测为异常才认为是异常
            outlier_count = outlier_mask_z.astype(int) + outlier_mask_iqr.astype(int) + outlier_mask_if.astype(int)
            outlier_mask = outlier_count >= 2
            outlier_indices = np.where(outlier_mask)[0]
        
        print(f"检测到 {len(outlier_indices)} 个异常点 (约 {len(outlier_indices)/len(df_a)*100:.2f}%)")
        
        # 保存异常值分布可视化
        plt.figure(figsize=(15, 10))
        
        # 1. 异常值占比饼图
        plt.subplot(2, 2, 1)
        plt.pie([len(df_a) - len(outlier_indices), len(outlier_indices)], 
                labels=['正常样本', '异常样本'],
                autopct='%1.1f%%')
        plt.title('异常值占比')
        
        # 2. 异常分布散点图 (随机选择2个特征)
        features = np.random.choice(df_a.columns, 2, replace=False)
        plt.subplot(2, 2, 2)
        plt.scatter(df_a[features[0]], df_a[features[1]], 
                   c=['red' if i in outlier_indices else 'blue' for i in range(len(df_a))],
                   alpha=0.5)
        plt.title(f'基于特征的异常值分布')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        
        # 3. 目标值异常散点图
        if df_b.shape[1] == 1:
            plt.subplot(2, 2, 3)
            target_values = df_b.iloc[:, 0].values
            plt.scatter(range(len(target_values)), target_values, 
                       c=['red' if i in outlier_indices else 'blue' for i in range(len(df_a))],
                       alpha=0.5)
            plt.title('目标变量异常分布')
            plt.xlabel('样本索引')
            plt.ylabel('目标值')
        
        # 4. 异常值处理方法说明
        plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.text(0.1, 0.5, f"异常检测方法: {outlier_method}\n"
                          f"异常点数量: {len(outlier_indices)}\n"
                          f"异常点比例: {len(outlier_indices)/len(df_a)*100:.2f}%\n"
                          f"处理方式:\n"
                          f"1. 添加异常标记特征\n"
                          f"2. 权重调整\n"
                          f"3. 鲁棒性训练",
                 fontsize=12)
        
        plt.tight_layout()
        plt.savefig('d:/amath/B/B1/results/异常值分析.png')
        plt.close()
        
        # 异常值处理策略：
        # 1. 为异常点添加标记特征
        df_a['is_outlier'] = 0
        df_a.loc[outlier_indices, 'is_outlier'] = 1
        
        # 2. 对异常点进行处理 - 可选择以下策略之一:
        # A. 替换为临近非异常点的中位数
        # B. 使用Winsorization (极值截断)
        # C. 平滑异常点
        
        # 这里采用平滑处理
        if len(outlier_indices) > 0:
            # 对B数据进行平滑处理
            if df_b.shape[1] == 1:
                # 提取并复制原始值
                original_values = df_b.iloc[:, 0].values.copy()
                
                # 对离群点应用Savitzky-Golay滤波器进行平滑
                window_size = min(21, len(df_b) - 1)
                if window_size % 2 == 0:
                    window_size -= 1  # 确保是奇数
                
                poly_order = min(3, window_size - 1)
                
                # 只对异常点应用平滑，保留正常点的原始值
                smoothed_values = savgol_filter(original_values, window_size, poly_order)
                
                # 只替换异常点的值
                original_values[outlier_indices] = smoothed_values[outlier_indices]
                
                # 更新B数据
                df_b.iloc[:, 0] = original_values
                
                # 可视化平滑效果
                plt.figure(figsize=(12, 6))
                plt.plot(range(len(original_values)), df_b.iloc[:, 0], 'b-', label='处理后')
                plt.plot(range(len(smoothed_values)), smoothed_values, 'g--', label='全部平滑')
                plt.scatter(outlier_indices, smoothed_values[outlier_indices], color='red', alpha=0.7, label='处理的异常点')
                plt.title('异常点平滑处理效果')
                plt.xlabel('样本索引')
                plt.ylabel('目标值')
                plt.legend()
                plt.savefig('d:/amath/B/B1/results/异常值平滑处理.png')
                plt.close()
    
    # 3. 检查并删除重复行 - 保持A和B的行对齐
    duplicates_a = df_a.duplicated().sum()
    duplicates_b = df_b.duplicated().sum()
    
    print(f"A数据重复行数量: {duplicates_a}")
    print(f"B数据重复行数量: {duplicates_b}")
    
    if duplicates_a > 0 or duplicates_b > 0:
        # 创建联合索引，标记重复行
        df_combined = pd.concat([df_a, df_b], axis=1)
        duplicated_rows = df_combined.duplicated()
        print(f"联合重复行数量: {duplicated_rows.sum()}")
        
        # 删除联合重复行
        df_combined = df_combined[~duplicated_rows]
        
        # 分割回A和B
        df_a = df_combined.iloc[:, :df_a.shape[1]]
        df_b = df_combined.iloc[:, df_a.shape[1]:]
        
        print(f"去重后: A数据({len(df_a)}行)和B数据({len(df_b)}行)")
    
    # 4. 确保A和B的行索引一致
    if len(df_a) != len(df_b):
        print(f"警告: 清洗后A数据({len(df_a)}行)和B数据({len(df_b)}行)行数不一致，进行强制对齐")
        # 重置索引以确保它们可以对齐
        df_a = df_a.reset_index(drop=True)
        df_b = df_b.reset_index(drop=True)
        
        # 取较小的行数
        min_rows = min(len(df_a), len(df_b))
        df_a = df_a.iloc[:min_rows]
        df_b = df_b.iloc[:min_rows]
        print(f"对齐后: A数据({len(df_a)}行)和B数据({len(df_b)}行)")
    
    # 5. 特征相关性分析
    if df_b.shape[1] == 1:
        b_values = df_b.iloc[:, 0]
        correlations = []
        
        for col in df_a.columns:
            if col != 'is_outlier':
                corr = df_a[col].corr(b_values)
                correlations.append((col, corr))
        
        # 按相关性排序
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 保存相关性结果
        corr_df = pd.DataFrame(correlations, columns=['特征', '相关性'])
        corr_df.to_csv('d:/amath/B/B1/results/特征相关性.csv', index=False)
        
        # 绘制前20个最相关特征的相关性图
        top_features = corr_df.head(20)
        plt.figure(figsize=(10, 8))
        plt.barh(top_features['特征'][::-1], top_features['相关性'][::-1])
        plt.title('与目标变量相关性最高的20个特征')
        plt.xlabel('相关性系数')
        plt.ylabel('特征名称')
        plt.savefig('d:/amath/B/B1/results/top20特征相关性.png')
        plt.close()
        
        print("已保存特征相关性分析结果")
    
    # 6. 添加额外的数据转换步骤
    df_a.columns = [f'feature_{i}' for i in range(df_a.shape[1] - 1)] + ['is_outlier']
    if df_b.shape[1] == 1:
        df_b.columns = ['target']
    
    print("数据清洗完成!")
    return df_a, df_b, outlier_indices

def preprocess_data(df_a, df_b, test_size=0.2, random_state=42):
    """
    增强版数据预处理:
    1. 使用稳健的缩放方法
    2. 提供异常样本的权重信息
    3. 准备用于FTTransformer的数据格式
    """
    # 使用RobustScaler代替StandardScaler，对异常值更稳健
    scaler = RobustScaler()
    a_scaled = scaler.fit_transform(df_a.drop('is_outlier', axis=1) if 'is_outlier' in df_a.columns else df_a)
    
    # 重新组合特征，包括异常标记
    if 'is_outlier' in df_a.columns:
        outlier_feature = df_a['is_outlier'].values.reshape(-1, 1)
        a_scaled = np.hstack((a_scaled, outlier_feature))
    
    # 将标准化后的数据转回DataFrame
    columns = list(df_a.drop('is_outlier', axis=1).columns) if 'is_outlier' in df_a.columns else list(df_a.columns)
    if 'is_outlier' in df_a.columns:
        columns.append('is_outlier')
    
    df_a_scaled = pd.DataFrame(a_scaled, columns=columns, index=df_a.index)
    
    # 获取B数据的值
    if isinstance(df_b, pd.DataFrame):
        df_b_values = df_b.copy()
    else:
        df_b_values = pd.DataFrame(df_b, columns=['target'])
    
    # 合并A和B数据
    combined_df = pd.concat([df_a_scaled, df_b_values], axis=1)
    
    # 分割数据集
    train_df, test_df = train_test_split(
        combined_df, test_size=test_size, random_state=random_state
    )
    
    # 分离特征和目标变量
    X_train = train_df[columns].values
    y_train = train_df[df_b_values.columns].values
    
    X_test = test_df[columns].values
    y_test = test_df[df_b_values.columns].values
    
    # 创建样本权重 - 异常点权重更低
    if 'is_outlier' in df_a.columns:
        # 最后一列是异常标记
        outlier_mask = X_train[:, -1] > 0
        
        # 设置权重 - 正常点权重为1，异常点权重为0.3
        weights = np.ones(len(X_train))
        weights[outlier_mask] = 0.3
        
        # 创建采样器
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
    else:
        sampler = None
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    if sampler:
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 保存数据集信息
    feature_info = {
        'continuous_cols': [col for col in columns if col != 'is_outlier'],
        'categorical_cols': ['is_outlier'] if 'is_outlier' in columns else [],
        'target_col': df_b_values.columns.tolist()
    }
    
    return train_loader, test_loader, scaler, feature_info
