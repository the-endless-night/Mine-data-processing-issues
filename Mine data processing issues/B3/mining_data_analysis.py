import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import matplotlib as mpl
import time
import os
from datetime import datetime

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加备用字体
plt.rcParams['axes.unicode_minus'] = False

# 创建一个函数用于处理特殊字符显示
def format_label(text):
    """处理特殊字符，确保在任何字体下都能正常显示"""
    # 将R²替换为R平方
    return text.replace('R²', 'R平方')

def load_data(x_file_path='3-X.xlsx', y_file_path='3-Y.xlsx'):
    """
    加载矿山监测数据
    
    参数:
    X: 从3-X.xlsx加载，每行包含多个特征值
    Y: 从3-Y.xlsx加载第一列数据
    注意：从第0行开始都是数据，没有表头
    
    返回:
    包含X和Y数据的字典
    """
    try:
        # 加载X数据 - 没有表头，所有列作为特征
        x_data = pd.read_excel(x_file_path, header=None)
        print(f"X数据加载成功，形状: {x_data.shape}")
        
        # 加载Y数据 - 没有表头，只取第一列
        y_data = pd.read_excel(y_file_path, header=None).iloc[:, 0]
        print(f"Y数据加载成功，形状: {y_data.shape}")
        
        # 确保数据具有相同的行数
        if len(x_data) != len(y_data):
            print(f"警告：X数据行数({len(x_data)})与Y数据行数({len(y_data)})不匹配")
            # 取最小长度
            min_len = min(len(x_data), len(y_data))
            x_data = x_data.iloc[:min_len]
            y_data = y_data.iloc[:min_len]
            print(f"已截取至共同长度: {min_len}行")
        
        print(f"数据处理成功，X特征数量: {x_data.shape[1]}, 样本数: {len(x_data)}")
        
        return {
            'X': x_data,
            'Y': y_data.values
        }
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def explore_data(data):
    """探索性数据分析"""
    x_data = data['X']
    y_data = data['Y']
    print("X数据前5行：")
    print(x_data.head())
    
    print("\nY数据前5个值：")
    print(y_data[:5])
    
    print("\nX数据基本统计信息：")
    print(x_data.describe())
    
    # 检查缺失值
    print("\n缺失值检查：")
    print(x_data.isnull().sum())
    
    # 绘制原始数据 - 显示前几个特征和Y值
    plt.figure(figsize=(15, 10))
    # 绘制X数据的前几个特征
    num_features_to_plot = min(3, x_data.shape[1])
    for i in range(num_features_to_plot):
        plt.subplot(num_features_to_plot+1, 1, i+1)
        plt.plot(x_data.iloc[:, i], label=f'特征{i+1}')
        plt.title(f'原始X数据 - 特征{i+1}')
        plt.legend()
    
    # 绘制Y数据
    plt.subplot(num_features_to_plot+1, 1, num_features_to_plot+1)
    plt.plot(y_data, label='Y数据', color='orange')
    plt.title('原始Y数据')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('原始数据.png')
    plt.close()

def check_and_remove_duplicates(data):
    """检查并处理重复数据"""
    x_data = data['X'].copy()
    y_data = data['Y'].copy()
    
    # 检查X数据中的重复行
    duplicate_rows = x_data.duplicated()
    duplicate_count = duplicate_rows.sum()
    if duplicate_count > 0:
        print(f"发现{duplicate_count}行重复数据 ({duplicate_count / len(x_data) * 100:.2f}%)")
        # 删除重复行
        x_clean = x_data.drop_duplicates()
        
        # 确保Y数据也相应更新
        y_clean = y_data[~duplicate_rows]
        
        print(f"删除重复数据后，数据形状: X={x_clean.shape}, Y=({len(y_clean)},)")
        # 返回清理后的数据
        return {
            'X': x_clean,
            'Y': y_clean
        }
    else:
        print("未发现重复数据")
        return data

def detect_and_handle_outliers(data, method='zscore', handling='winsorize'):
    """
    检测并处理异常值
    
    参数:
    data: 数据字典，包含'X'和'Y'
    method: 异常值检测方法，可选'zscore'、'iqr'
    handling: 异常值处理方法，可选'winsorize'、'remove'、'mean'、'median'
    
    返回:
    清理后的数据字典
    """
    x_data = data['X'].copy()
    y_data = data['Y'].copy()
    
    # 对X数据进行异常值检测和处理
    outliers_info = {}
    x_cleaned = pd.DataFrame(index=x_data.index, columns=x_data.columns)
    for col in x_data.columns:
        # 获取当前列数据
        col_data = x_data[col]
        
        # 检测异常值
        if method == 'zscore':
            # 使用Z-score方法（|z| > 3为异常值）
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outliers = z_scores > 3
            outlier_indices = np.where(outliers)[0]
        elif method == 'iqr':
            # 使用IQR方法（Q1 - 1.5*IQR 或 Q3 + 1.5*IQR为异常值）
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_indices = np.where(outliers)[0]
        
        # 记录异常值信息
        outliers_info[col] = {
            '异常值数量': len(outlier_indices),
            '异常值百分比': len(outlier_indices) / len(col_data) * 100
        }
        
        # 处理异常值
        col_cleaned = col_data.copy()
        if handling == 'winsorize':
            # 截断法 - 将异常值替换为边界值
            if method == 'zscore':
                mean, std = col_data.mean(), col_data.std()
                col_cleaned[outliers] = np.clip(col_data[outliers], mean - 3*std, mean + 3*std)
            elif method == 'iqr':
                col_cleaned[outliers] = np.clip(col_data[outliers], lower_bound, upper_bound)
        elif handling == 'mean':
            # 均值替换
            col_cleaned[outliers] = col_data.mean()
        elif handling == 'median':
            # 中位数替换
            col_cleaned[outliers] = col_data.median()
        elif handling == 'remove':
            # 移除异常值（后续会过滤整行）
            col_cleaned[outliers] = np.nan
        
        # 保存处理后的数据
        x_cleaned[col] = col_cleaned
    
    # 如果采用remove策略，需要过滤掉包含NaN的行
    if handling == 'remove':
        mask = ~x_cleaned.isna().any(axis=1)
        x_cleaned = x_cleaned[mask]
        y_cleaned = y_data[mask]
    else:
        y_cleaned = y_data
    
    # 打印异常值检测和处理的汇总信息
    print(f"\n异常值检测 (方法: {method}):")
    total_outliers = sum(info['异常值数量'] for info in outliers_info.values())
    if total_outliers > 0:
        print(f"共检测到{total_outliers}个异常值点")
        for col, info in outliers_info.items():
            if info['异常值数量'] > 0:
                print(f"  特征{col}：{info['异常值数量']}个 ({info['异常值百分比']:.2f}%)")
        print(f"\n异常值处理 (方法: {handling}):")
        if handling == 'remove':
            print(f"移除包含异常值的行，剩余样本数: {len(x_cleaned)}")
        else:
            print(f"异常值已被替换处理，数据形状保持不变: {x_cleaned.shape}")
    else:
        print("未检测到异常值")
    
    # 可视化异常值处理效果
    visualize_outlier_handling(x_data, x_cleaned, outliers_info, method, handling)
    
    return {
        'X': x_cleaned,
        'Y': y_cleaned
    }

def visualize_outlier_handling(x_original, x_cleaned, outliers_info, method, handling):
    """可视化异常值处理前后的对比"""
    # 选择前3个具有最多异常值的特征
    most_outliers_cols = sorted(outliers_info.keys(), 
                              key=lambda col: outliers_info[col]['异常值数量'], 
                              reverse=True)[:3]
    
    # 如果没有异常值，返回
    if all(outliers_info[col]['异常值数量'] == 0 for col in most_outliers_cols):
        return
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    # 为每个特征创建一个子图
    for i, col in enumerate(most_outliers_cols):
        if outliers_info[col]['异常值数量'] == 0:
            continue
        plt.subplot(len(most_outliers_cols), 1, i+1)
        
        # 绘制原始数据
        plt.plot(x_original[col], 'b.', alpha=0.5, label='原始数据')
        # 绘制处理后的数据
        plt.plot(x_cleaned[col], 'r.', alpha=0.5, label='清理后')
        
        # 添加标题和标签
        plt.title(f'特征{col}异常值处理 - {outliers_info[col]["异常值数量"]}个异常值 ({outliers_info[col]["异常值百分比"]:.2f}%)')
        plt.ylabel('值')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'异常值处理效果_{method}_{handling}.png')
    plt.close()

def impute_missing_values(data, method='median'):
    """
    填充缺失值
    
    参数:
    data: 数据字典，包含'X'和'Y'
    method: 填充方法，可选'mean'、'median'、'mode'、'knn'
    
    返回:
    填充后的数据字典
    """
    x_data = data['X'].copy()
    y_data = data['Y'].copy()
    
    # 检查缺失值
    missing_count = x_data.isnull().sum()
    total_missing = missing_count.sum()
    if total_missing > 0:
        print(f"\n发现{total_missing}个缺失值:")
        for col, count in missing_count[missing_count > 0].items():
            print(f"  特征{col}: {count}个 ({count / len(x_data) * 100:.2f}%)")
        
        # 填充缺失值
        if method == 'mean':
            x_filled = x_data.fillna(x_data.mean())
            print("使用特征均值填充缺失值")
        elif method == 'median':
            x_filled = x_data.fillna(x_data.median())
            print("使用特征中位数填充缺失值")
        elif method == 'mode':
            x_filled = x_data.fillna(x_data.mode().iloc[0])
            print("使用特征众数填充缺失值")
        elif method == 'knn':
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            x_filled = pd.DataFrame(
                imputer.fit_transform(x_data),
                columns=x_data.columns,
                index=x_data.index
            )
            print("使用KNN方法填充缺失值")
        else:
            x_filled = x_data.fillna(x_data.median())
            print("默认使用特征中位数填充缺失值")
    else:
        print("数据中没有缺失值")
        x_filled = x_data
    
    return {
        'X': x_filled,
        'Y': y_data
    }

def data_cleaning(data):
    """
    数据清理主函数，整合所有清理步骤
    """
    print("\n===== 开始数据清理 =====")
    
    # 步骤1: 处理重复值
    data = check_and_remove_duplicates(data)
    
    # 步骤2: 填充缺失值
    data = impute_missing_values(data, method='median')
    
    # 步骤3: 处理异常值
    data = detect_and_handle_outliers(data, method='zscore', handling='winsorize')
    
    print("===== 数据清理完成 =====\n")
    return data

def denoise_data(data, method='savgol'):
    """
    对X数据的每个特征进行去噪处理
    """
    x_data = data['X']
    x_denoised = pd.DataFrame()
    for col in x_data.columns:
        col_data = x_data[col].values
        
        if method == 'savgol':
            # Savitzky-Golay滤波
            window_length = min(11, len(col_data) - 2)  # 确保窗口长度小于数据长度
            # 确保窗口长度是奇数
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(3, window_length - 1)  # 多项式阶数必须小于窗口长度
            denoised = signal.savgol_filter(col_data, window_length, polyorder)
            method_name = 'Savitzky-Golay滤波'
        elif method == 'moving_avg':
            # 移动平均滤波
            window_size = min(5, len(col_data))
            denoised = np.convolve(col_data, np.ones(window_size)/window_size, mode='same')
            method_name = '移动平均滤波'
            
        elif method == 'median':
            # 中值滤波
            window_size = min(5, len(col_data))
            denoised = signal.medfilt(col_data, kernel_size=window_size)
            method_name = '中值滤波'
            
        x_denoised[col] = denoised
    
    # 可视化前几个特征的去噪效果
    plt.figure(figsize=(15, 10))
    num_features_to_plot = min(3, x_data.shape[1])
    for i, col in enumerate(x_data.columns[:num_features_to_plot]):
        plt.subplot(num_features_to_plot, 1, i+1)
        plt.plot(x_data[col], label='原始数据', alpha=0.7)
        plt.plot(x_denoised[col], label=f'{method_name}后', linewidth=2)
        plt.title(f'特征{i+1}去噪效果对比 ({method_name})')
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'X数据去噪效果_{method}.png')
    plt.close()
    
    return x_denoised

def standardize_data(x_data):
    """对X数据的每个特征进行标准化处理"""
    scaler = StandardScaler()
    x_standardized = pd.DataFrame(
        scaler.fit_transform(x_data),
        columns=x_data.columns
    )
    # 可视化前几个特征的标准化效果
    plt.figure(figsize=(15, 10))
    num_features_to_plot = min(3, x_data.shape[1])
    for i, col in enumerate(x_data.columns[:num_features_to_plot]):
        plt.subplot(num_features_to_plot, 1, i+1)
        plt.plot(x_data[col], label='去噪后数据')
        plt.plot(x_standardized[col], label='标准化后数据')
        plt.title(f'特征{i+1}标准化效果对比')
        plt.legend()
    plt.tight_layout()
    plt.savefig('X数据标准化效果.png')
    plt.close()
    
    return x_standardized, scaler

def create_models():
    """创建多种回归模型"""
    models = {
        "线性回归": LinearRegression(),
        "岭回归": Ridge(alpha=1.0),
        "Lasso回归": Lasso(alpha=0.1),
        "弹性网络回归": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "决策树回归": DecisionTreeRegressor(max_depth=5),
        "随机森林回归": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "梯度提升回归": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "支持向量回归": SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
    }
    return models

def train_and_evaluate_model(model, X_train, y_train, X_test=None, y_test=None, model_name="未命名模型"):
    """训练模型并评估性能"""
    start_time = time.time()
    # 如果没有提供测试集，使用训练集进行评估
    if X_test is None or y_test is None:
        X_test, y_test = X_train, y_train
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 计算训练集和测试集的评估指标
    train_metrics = {
        'R²': r2_score(y_train, y_pred_train),
        'MSE': mean_squared_error(y_train, y_pred_train),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'MAE': mean_absolute_error(y_train, y_pred_train)
    }
    test_metrics = {
        'R²': r2_score(y_test, y_pred_test),
        'MSE': mean_squared_error(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'MAE': mean_absolute_error(y_test, y_pred_test)
    }
    
    # 进行F检验（回归显著性检验）
    n = len(y_test)  # 样本数
    k = X_test.shape[1]  # 特征数量
    # 计算回归平方和（SSR）
    y_mean = np.mean(y_test)
    ssr = np.sum((y_pred_test - y_mean) ** 2)
    # 计算残差平方和（SSE）
    sse = np.sum((y_test - y_pred_test) ** 2)
    # 计算总平方和（SST）
    sst = np.sum((y_test - y_mean) ** 2)
    
    # 计算F统计量
    f_statistic = (ssr / k) / (sse / (n - k - 1)) if sse > 0 and n > k + 1 else 0
    
    # 计算p值
    p_value = 1 - stats.f.cdf(f_statistic, k, n - k - 1) if f_statistic > 0 else 1
    
    test_metrics.update({
        'F统计量': f_statistic,
        'p值': p_value
    })
    
    # 计算训练时间
    training_time = time.time() - start_time
    
    # 计算交叉验证得分
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_r2 = np.mean(cv_scores)
    except:
        cv_r2 = None
    
    result = {
        'model': model,
        'model_name': model_name,
        'training_time': training_time,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'cv_r2': cv_r2
    }
    return result

def compare_models(models_results, top_n=None):
    """比较多个模型的性能"""
    # 创建评估指标的数据框
    results_df = pd.DataFrame({
        '模型': [result['model_name'] for result in models_results],
        '训练集R²': [result['train_metrics']['R²'] for result in models_results],
        '测试集R²': [result['test_metrics']['R²'] for result in models_results],
        '交叉验证R²': [result['cv_r2'] if result['cv_r2'] is not None else 0 for result in models_results],
        '训练集RMSE': [result['train_metrics']['RMSE'] for result in models_results],
        '测试集RMSE': [result['test_metrics']['RMSE'] for result in models_results],
        '训练时间(秒)': [result['training_time'] for result in models_results],
        'F统计量': [result['test_metrics']['F统计量'] for result in models_results],
        'p值': [result['test_metrics']['p值'] for result in models_results]
    })
    
    # 按测试集R²降序排序，如果有交叉验证，则优先使用交叉验证R²
    if all(results_df['交叉验证R²'] > 0):
        results_df = results_df.sort_values(by='交叉验证R²', ascending=False)
    else:
        results_df = results_df.sort_values(by='测试集R²', ascending=False)
    
    # 如果指定了top_n，则只保留前top_n个模型
    if top_n is not None and top_n < len(results_df):
        results_df = results_df.head(top_n)
    
    return results_df

def visualize_model_comparison(models_results):
    """可视化模型比较结果"""
    # 提取模型名称和评估指标
    model_names = [result['model_name'] for result in models_results]
    train_r2 = [result['train_metrics']['R²'] for result in models_results]
    test_r2 = [result['test_metrics']['R²'] for result in models_results]
    cv_r2 = [result['cv_r2'] if result['cv_r2'] is not None else 0 for result in models_results]
    train_rmse = [result['train_metrics']['RMSE'] for result in models_results]
    test_rmse = [result['test_metrics']['RMSE'] for result in models_results]
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 绘制R²对比 - 使用format_label处理标签
    bar_width = 0.25
    x = np.arange(len(model_names))
    
    axes[0].bar(x - bar_width, train_r2, bar_width, label=format_label('训练集R²'), color='skyblue')
    axes[0].bar(x, test_r2, bar_width, label=format_label('测试集R²'), color='orange')
    axes[0].bar(x + bar_width, cv_r2, bar_width, label=format_label('交叉验证R²'), color='green')
    
    axes[0].set_xlabel('模型')
    axes[0].set_ylabel(format_label('R²值'))
    axes[0].set_title(format_label('不同模型的R²对比'))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 绘制RMSE对比
    axes[1].bar(x - bar_width/2, train_rmse, bar_width, label='训练集RMSE', color='skyblue')
    axes[1].bar(x + bar_width/2, test_rmse, bar_width, label='测试集RMSE', color='orange')
    
    axes[1].set_xlabel('模型')
    axes[1].set_ylabel('RMSE值')
    axes[1].set_title('不同模型的RMSE对比')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('模型性能对比.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_best_model(best_result, X_test, y_test):
    """可视化最佳模型的预测结果"""
    model_name = best_result['model_name']
    y_pred = best_result['y_pred_test']
    metrics = best_result['test_metrics']
    
    plt.figure(figsize=(12, 10))
    
    # 绘制预测值与实际值对比图
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(y_test)), y_test, label='实际值', alpha=0.7, color='blue')
    plt.plot(range(len(y_pred)), y_pred, 'r-', label='预测值', linewidth=2)
    
    # 在图上直接标注R平方值
    r2_text = f"{format_label('R²')} = {metrics['R²']:.4f}"
    plt.annotate(r2_text, xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                fontsize=12)
    
    plt.title(f'最佳模型 ({model_name}) 的预测效果')
    plt.legend(loc='upper right')
    
    # 绘制残差图
    residuals = y_test - y_pred
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(residuals)), residuals, color='green')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('残差分布')
    plt.xlabel('样本索引')
    plt.ylabel('残差')
    
    # 添加评估指标文本
    metrics_text = ""
    for k, v in metrics.items():
        if k != 'R²':  # R平方已在上图中显示
            metrics_text += f"{k}: {v:.4f}\n"
    
    plt.figtext(0.5, 0.01, metrics_text, 
                ha="center", fontsize=10, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'最佳模型_{model_name}_预测效果.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_residual_analysis_plot(best_result, y_test):
    """创建残差分析图"""
    y_pred = best_result['y_pred_test']
    model_name = best_result['model_name']
    residuals = y_test - y_pred
    
    plt.figure(figsize=(14, 10))
    
    # 1. 残差散点图
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差vs预测值')
    
    # 2. 残差直方图
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.xlabel('残差')
    plt.ylabel('频数')
    plt.title('残差分布直方图')
    
    # 3. QQ图检验残差的正态性
    plt.subplot(2, 2, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('残差正态Q-Q图')
    
    # 4. 残差自相关图
    plt.subplot(2, 2, 4)
    if len(residuals) > 1:
        lag_autocorrelation = [np.corrcoef(residuals[:-i], residuals[i:])[0, 1] for i in range(1, min(11, len(residuals)))]
        plt.bar(range(1, len(lag_autocorrelation) + 1), lag_autocorrelation)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('滞后期数')
        plt.ylabel('自相关系数')
        plt.title('残差自相关图')
    else:
        plt.text(0.5, 0.5, '样本数量不足，无法计算自相关', ha='center', va='center')
        plt.title('残差自相关图 (无法计算)')
    
    plt.suptitle(f'模型 ({model_name}) 残差分析', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'残差分析_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_model_explanation(best_result, feature_names=None):
    """生成模型解释报告"""
    model = best_result['model']
    model_name = best_result['model_name']
    metrics = best_result['test_metrics']
    
    report = f"## {model_name} 模型解释\n\n"
    
    # 添加模型性能指标
    report += "### 模型性能指标\n\n"
    report += "| 指标 | 值 |\n"
    report += "| --- | --- |\n"
    for metric, value in metrics.items():
        metric_name = format_label("R²") if metric == "R²" else metric
        report += f"| {metric_name} | {value:.4f} |\n"
    report += "\n"
    
    # 如果是线性模型，添加系数解释
    if hasattr(model, 'coef_') and feature_names is not None:
        report += "### 特征重要性/系数\n\n"
        report += "| 特征 | 系数 |\n"
        report += "| --- | --- |\n"
        
        if model_name in ["线性回归", "岭回归", "Lasso回归", "弹性网络回归"]:
            # 处理线性模型的系数
            coefs = model.coef_
            intercept = model.intercept_
            
            # 打印截距
            report += f"| 截距 | {intercept:.4f} |\n"
            
            # 打印各特征系数
            for feature, coef in zip(feature_names, coefs):
                report += f"| {feature} | {coef:.4f} |\n"
                
            report += "\n#### 模型方程\n\n"
            equation = f"Y = {intercept:.4f}"
            
            for feature, coef in zip(feature_names, coefs):
                sign = "+" if coef >= 0 else ""
                equation += f" {sign} {coef:.4f} × {feature}"
                
            report += f"`{equation}`\n\n"
    
    # 如果是树模型，添加特征重要性
    elif hasattr(model, 'feature_importances_') and feature_names is not None:
        report += "### 特征重要性\n\n"
        report += "| 特征 | 重要性 |\n"
        report += "| --- | --- |\n"
        
        # 获取特征重要性
        importances = model.feature_importances_
        
        # 按重要性降序排序
        indices = np.argsort(importances)[::-1]
        
        for i in indices:
            report += f"| {feature_names[i]} | {importances[i]:.4f} |\n"
    
    # 添加解释性文字
    report += "\n### 模型解释\n\n"
    
    if model_name == "线性回归":
        report += "线性回归模型假设自变量与因变量之间存在线性关系。每个系数表示在其他变量保持不变的情况下，该特征每增加一个单位，目标变量的预期变化量。\n\n"
    elif model_name == "岭回归":
        report += "岭回归是一种带L2正则化的线性回归。相比普通线性回归，它在处理多重共线性时表现更好，可以有效减少过拟合风险。\n\n"
    elif model_name == "Lasso回归":
        report += "Lasso回归是一种带L1正则化的线性回归，具有特征选择能力。非零系数对应的特征是模型认为重要的特征，系数为零的特征被模型认为不重要。\n\n"
    elif model_name == "弹性网络回归":
        report += "弹性网络回归结合了岭回归和Lasso回归的优点，同时使用L1和L2正则化。它在存在多个相关特征的情况下表现良好。\n\n"
    elif model_name == "决策树回归":
        report += "决策树通过一系列规则将数据分割成子集，对每个数据子集给出单独的预测。特征重要性反映了该特征在降低预测误差方面的贡献。\n\n"
    elif model_name == "随机森林回归":
        report += "随机森林通过平均多个决策树的预测结果来提高预测准确性和控制过拟合。特征重要性表示该特征在所有树中的平均贡献度。\n\n"
    elif model_name == "梯度提升回归":
        report += "梯度提升是一种强大的集成方法，通过串联多个弱学习器来构建强预测模型。特征重要性反映了每个特征对模型性能的总体贡献。\n\n"
    elif model_name == "支持向量回归":
        report += "支持向量回归通过确定最佳超平面来预测连续值，其目标是最小化预测误差，同时保持预测的裕度尽可能大。\n\n"
    
    # 添加统计显著性分析
    report += "### 统计显著性分析\n\n"
    p_value = metrics['p值']
    f_statistic = metrics['F统计量']
    
    report += f"F统计量: {f_statistic:.4f}\n\n"
    report += f"p值: {p_value:.4f}\n\n"
    
    if p_value < 0.05:
        report += "**结论**: 在0.05的显著性水平下，该模型具有统计显著性，模型可靠地解释了因变量的变异。\n\n"
    elif p_value < 0.1:
        report += "**结论**: 在0.1的显著性水平下，该模型具有统计显著性，但解释能力相对较弱。\n\n"
    else:
        report += "**结论**: 该模型不具有统计显著性，可能无法可靠地解释因变量的变异。\n\n"
    
    return report

def generate_analysis_report(data, models_results, best_result, pca=None):
    """生成完整的分析报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"# 矿山监测数据分析报告\n\n"
    report += f"**生成时间**: {now}\n\n"
    
    # 1. 数据概览
    report += "## 1. 数据概览\n\n"
    x_data = data['X']
    y_data = data['Y']
    
    report += f"### 1.1 数据集信息\n\n"
    report += f"- X数据形状: {x_data.shape}\n"
    report += f"- Y数据形状: {y_data.shape}\n\n"
    
    # 2. 数据预处理
    report += "## 2. 数据预处理方法\n\n"
    
    report += "### 2.1 去噪处理\n\n"
    report += "使用Savitzky-Golay滤波器进行去噪处理，该方法的优势：\n\n"
    report += "- 能够有效平滑数据并保留原始信号的形态特征\n"
    report += "- 通过局部多项式拟合减少噪声影响\n"
    report += "- 在保持数据峰值和趋势的同时去除高频噪声\n\n"
    
    report += "**参数设置**：\n"
    report += "- 窗口长度：自适应设置 (至少为奇数)\n"
    report += "- 多项式阶数：3\n\n"
    
    report += "### 2.2 标准化处理\n\n"
    report += "采用Z-score标准化，将数据转换为均值为0、标准差为1的分布：\n\n"
    report += "\\[ X_{标准化} = \\frac{X - \\mu}{\\sigma} \\]\n\n"
    report += "其中，$\\mu$为X的均值，$\\sigma$为X的标准差。\n\n"
    report += "标准化的优势：\n\n"
    report += "- 消除不同特征的量纲影响\n"
    report += "- 使不同特征具有可比性\n"
    report += "- 提高模型的收敛速度和稳定性\n\n"
    
    # 3. 模型比较
    report += "## 3. 模型比较\n\n"
    
    # 创建模型比较表格
    model_comparison_df = compare_models(models_results)
    
    # 手动创建Markdown表格
    report += "| 模型 | 训练集R平方 | 测试集R平方 | 交叉验证R平方 | 训练集RMSE | 测试集RMSE | 训练时间(秒) | F统计量 | p值 |\n"
    report += "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for _, row in model_comparison_df.iterrows():
        report += "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            row['模型'],
            row['训练集R²'],
            row['测试集R²'],
            row['交叉验证R²'],
            row['训练集RMSE'],
            row['测试集RMSE'],
            row['训练时间(秒)'],
            row['F统计量'],
            row['p值']
        )
    
    report += "\n"
    
    # 4. 最佳模型分析
    report += "## 4. 最佳模型分析\n\n"
    
    # 获取最佳模型信息
    best_model_name = best_result['model_name']
    best_metrics = best_result['test_metrics']
    
    report += f"### 4.1 最佳模型: {best_model_name}\n\n"
    
    # 添加模型性能指标
    report += "**性能指标**:\n\n"
    report += "| 指标 | 值 |\n"
    report += "| --- | --- |\n"
    for metric, value in best_metrics.items():
        metric_name = format_label("R²") if metric == "R²" else metric
        report += f"| {metric_name} | {value:.4f} |\n"
    report += "\n"
    
    # 生成特征名称
    feature_names = [f"特征{i+1}" for i in range(x_data.shape[1])]
    
    # 添加模型解释
    model_explanation = generate_model_explanation(best_result, feature_names)
    report += model_explanation
    
    # 5. 拟合优度计算过程
    report += "## 5. 拟合优度计算过程\n\n"
    
    report += "### 5.1 决定系数(R平方)\n\n"
    report += "决定系数表示模型解释的因变量变异比例，计算公式：\n\n"
    report += "\\[ R^2 = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2} = 1 - \\frac{SSE}{SST} \\]\n\n"
    report += "其中，$y_i$是实际值，$\\hat{y}_i$是预测值，$\\bar{y}$是实际值的均值。\n\n"
    report += "- SSE: 残差平方和\n"
    report += "- SST: 总平方和\n\n"
    
    report += "### 5.2 均方误差(MSE)\n\n"
    report += "均方误差是预测值与实际值差的平方和的平均值：\n\n"
    report += "\\[ MSE = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2 \\]\n\n"
    
    report += "### 5.3 均方根误差(RMSE)\n\n"
    report += "均方根误差是MSE的平方根，与因变量具有相同量纲：\n\n"
    report += "\\[ RMSE = \\sqrt{MSE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2} \\]\n\n"
    
    report += "### 5.4 平均绝对误差(MAE)\n\n"
    report += "平均绝对误差是预测值与实际值绝对差的平均值：\n\n"
    report += "\\[ MAE = \\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i| \\]\n\n"
    
    report += "### 5.5 F检验\n\n"
    report += "F检验用于评估模型的整体显著性，计算公式：\n\n"
    report += "\\[ F = \\frac{SSR/k}{SSE/(n-k-1)} \\]\n\n"
    report += "其中，SSR为回归平方和，SSE为残差平方和，n为样本数，k为自变量数量。\n\n"
    report += "- p值越小，表明模型越显著\n"
    report += "- 通常当p < 0.05时，认为模型具有统计显著性\n\n"
    
    # 6. 误差分析
    report += "## 6. 误差分析\n\n"
    
    report += "### 6.1 残差分析\n\n"
    report += "残差是实际值与预测值之间的差异：$e_i = y_i - \\hat{y}_i$\n\n"
    report += "有效的模型应该满足以下残差假设：\n\n"
    report += "1. **独立性**：残差之间应相互独立，没有明显的模式\n"
    report += "2. **正态性**：残差应近似服从正态分布\n"
    report += "3. **同方差性**：残差的方差应该在预测值范围内保持恒定\n"
    report += "4. **线性性**：残差与预测值之间不应存在非线性关系\n\n"
    
    report += "### 6.2 模型诊断\n\n"
    report += "通过以下图表进行模型诊断：\n\n"
    report += "1. **残差-预测值散点图**：检查同方差性和线性性\n"
    report += "2. **残差直方图**：检查残差的分布是否接近正态分布\n"
    report += "3. **Q-Q图**：比较残差的分布与标准正态分布\n"
    report += "4. **残差自相关图**：检查残差之间是否存在自相关\n\n"
    
    # 7. 结论
    report += "## 7. 结论\n\n"
    
    # 根据最佳模型的评估结果生成结论
    r2 = best_metrics['R²']
    p_value = best_metrics['p值']
    
    report += f"- 最佳模型为 **{best_model_name}**，拟合优度({format_label('R²')})为 **{r2:.4f}**\n"
    
    if r2 > 0.8:
        r2_quality = "优秀"
    elif r2 > 0.6:
        r2_quality = "良好"
    elif r2 > 0.4:
        r2_quality = "一般"
    else:
        r2_quality = "较差"
    
    report += f"- 拟合优度水平: **{r2_quality}**\n"
    
    if p_value < 0.05:
        report += f"- 模型具有统计显著性 (p值 = {p_value:.4f} < 0.05)\n"
    else:
        report += f"- 模型不具有统计显著性 (p值 = {p_value:.4f} >= 0.05)\n"
    
    # 根据最佳模型类型给出特定结论
    if best_model_name in ["线性回归", "岭回归", "Lasso回归", "弹性网络回归"]:
        report += "- 线性模型可以有效解释X与Y之间的关系，说明数据具有较好的线性结构\n"
    elif best_model_name in ["决策树回归", "随机森林回归", "梯度提升回归"]:
        report += "- 树模型表现最佳，说明X与Y之间可能存在复杂的非线性关系\n"
    elif best_model_name == "支持向量回归":
        report += "- 支持向量回归表现最佳，说明数据可能存在较复杂的非线性结构，且对异常值较敏感\n"
    
    report += "\n对于进一步改进模型性能的建议：\n\n"
    report += "1. 收集更多样本数据，增加模型的学习能力\n"
    report += "2. 探索更多特征工程方法，提取更有信息量的特征\n"
    report += "3. 进一步调整模型超参数，优化模型性能\n"
    report += "4. 考虑使用更复杂的模型或集成多个模型的预测结果\n"
    
    return report

def save_report_to_file(report, filename="矿山监测数据分析报告.md"):
    """将报告保存到文件中"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"分析报告已保存到文件：{filename}")

def multiple_train_test_splits(X, y, models, n_splits=5, test_size=0.2, random_state=None):
    """
    执行多次随机划分的训练和评估
    
    参数:
    X: 特征数据
    y: 目标变量
    models: 模型字典
    n_splits: 随机划分次数
    test_size: 测试集比例
    random_state: 随机种子，None表示完全随机划分
    
    返回:
    all_models_results: 所有模型在所有划分上的结果
    avg_models_results: 所有模型的平均性能
    """
    all_models_results = []
    
    for split in range(n_splits):
        print(f"\n执行随机划分 {split+1}/{n_splits}")
        
        # 随机划分数据
        if random_state is not None:
            # 使用不同的随机种子，但仍然可重现
            current_seed = random_state + split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=current_seed
            )
            print(f"使用随机种子: {current_seed}")
        else:
            # 完全随机划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size
            )
            print("使用完全随机划分")
        
        print(f"训练集形状: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"测试集形状: X_test {X_test.shape}, y_test {y_test.shape}")
        
        # 训练和评估模型
        split_results = []
        for model_name, model in models.items():
            print(f"训练和评估 {model_name}...")
            try:
                result = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
                split_results.append(result)
                print(f"  R平方 训练集: {result['train_metrics']['R²']:.4f}, 测试集: {result['test_metrics']['R²']:.4f}")
            except Exception as e:
                print(f"  模型 {model_name} 训练失败: {str(e)}")
        
        all_models_results.append(split_results)
    
    # 计算平均性能
    avg_models_results = []
    if n_splits > 1:
        print("\n计算多次随机划分的平均性能...")
        
        # 获取所有模型名称
        model_names = [model for model in models.keys()]
        
        # 对每个模型计算平均性能
        for model_idx, model_name in enumerate(model_names):
            # 收集该模型在所有划分上的结果
            model_results = []
            for split_results in all_models_results:
                for result in split_results:
                    if result['model_name'] == model_name:
                        model_results.append(result)
            
            if not model_results:
                continue
            
            # 创建一个平均结果
            avg_result = model_results[0].copy()
            
            # 计算平均训练集指标
            avg_train_metrics = {}
            for metric in model_results[0]['train_metrics']:
                avg_train_metrics[metric] = np.mean([r['train_metrics'][metric] for r in model_results])
            
            # 计算平均测试集指标
            avg_test_metrics = {}
            for metric in model_results[0]['test_metrics']:
                avg_test_metrics[metric] = np.mean([r['test_metrics'][metric] for r in model_results])
            
            # 计算平均训练时间
            avg_training_time = np.mean([r['training_time'] for r in model_results])
            
            # 更新平均结果
            avg_result['train_metrics'] = avg_train_metrics
            avg_result['test_metrics'] = avg_test_metrics
            avg_result['training_time'] = avg_training_time
            avg_result['model_name'] = f"{model_name} (平均{n_splits}次)"
            
            avg_models_results.append(avg_result)
            
            print(f"{model_name} 平均性能 - R平方: {avg_test_metrics['R²']:.4f}, RMSE: {avg_test_metrics['RMSE']:.4f}")
    
    return all_models_results, avg_models_results if n_splits > 1 else all_models_results[0]

def main():
    # 1. 加载数据
    data = load_data('3-X.xlsx', '3-Y.xlsx')
    if data is None:
        return
    
    # 2. 探索性数据分析
    explore_data(data)
    
    # 2.5 数据清理（新增步骤）
    data = data_cleaning(data)
    
    # 3. 数据预处理
    # 3.1 去噪处理
    x_denoised = denoise_data(data, method='savgol')
    
    # 3.2 标准化处理
    x_standardized, scaler = standardize_data(x_denoised)
    
    # 设置随机划分参数
    n_splits = 5  # 随机划分次数
    test_size = 0.2  # 测试集比例
    random_state = None  # 设置为None表示完全随机划分，设置为整数表示可重现的随机划分
    
    # 创建多种模型
    models = create_models()
    
    # 执行多次随机划分的训练和评估
    all_results, models_results = multiple_train_test_splits(
        x_standardized, data['Y'], models, 
        n_splits=n_splits, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # 比较模型性能
    comparison_df = compare_models(models_results)
    print("\n模型性能比较 (多次随机划分平均):")
    print(comparison_df)
    
    # 可视化模型比较
    visualize_model_comparison(models_results)
    
    # 选择最佳模型
    best_result = models_results[comparison_df.index[0]]
    best_model_name = best_result['model_name']
    print(f"\n最佳模型: {best_model_name}")
    
    # 对最佳模型类型进行最终评估
    # 取出原始模型名称（去掉"平均n次"后缀）
    original_model_name = best_model_name.split(" (平均")[0]
    final_model = models[original_model_name]
    
    # 使用全部数据进行最终训练
    print(f"\n使用全部数据训练最终模型: {original_model_name}")
    X_full = x_standardized
    y_full = data['Y']
    final_result = train_and_evaluate_model(final_model, X_full, y_full, model_name=original_model_name)
    
    # 10. 可视化最佳模型
    visualize_best_model(final_result, X_full, y_full)
    
    # 11. 残差分析
    create_residual_analysis_plot(final_result, y_full)
    
    # 12. 生成完整的分析报告
    report = generate_analysis_report(data, models_results, final_result)
    
    # 13. 保存报告到文件
    save_report_to_file(report)
    
    # 14. 打印总结性结论
    print("\n========== 分析总结 ==========")
    print(f"最佳模型: {original_model_name}")
    print(f"拟合优度({format_label('R²')}): {final_result['test_metrics']['R²']:.4f}")
    print(f"RMSE: {final_result['test_metrics']['RMSE']:.4f}")
    if final_result['test_metrics']['p值'] < 0.05:
        print("统计检验结果: 模型具有统计显著性 (p < 0.05)")
    else:
        print("统计检验结果: 模型不具有统计显著性 (p >= 0.05)")
    print(f"F统计量: {final_result['test_metrics']['F统计量']:.4f}")
    print(f"p值: {final_result['test_metrics']['p值']:.4f}")
    print("\n完整分析报告已保存到文件: 矿山监测数据分析报告.md")

if __name__ == "__main__":
    main()
