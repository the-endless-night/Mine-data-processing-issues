import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
def evaluate_model(model, test_loader, device):
    """
    评估模型性能并分析误差
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, batch_targets in test_loader:
            inputs, batch_targets = inputs.to(device), batch_targets.to(device)
            batch_predictions = model(inputs)
            
            predictions.append(batch_predictions.cpu().numpy())
            targets.append(batch_targets.cpu().numpy())
    
    # 将预测和目标转换为NumPy数组
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # 计算误差指标
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"R² 分数: {r2:.4f}")
    
    # 绘制预测vs真实值的散点图
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 真实值')
    plt.savefig('d:/amath/B/B1/results/预测散点图.png')
    plt.close()
    
    # 绘制误差分布直方图
    errors = predictions - targets
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.xlabel('误差')
    plt.ylabel('频率')
    plt.title('误差分布')
    plt.savefig('d:/amath/B/B1/results/误差分布.png')
    plt.close()
    
    # 分析误差来源
    analyze_errors(errors, predictions, targets)
    
    return predictions, targets, mse, rmse, mae, r2

def analyze_errors(errors, predictions, targets):
    """
    分析误差来源:
    1. 数据噪声
    2. 模型偏差
    3. 异常值检测
    """
    # 计算误差基本统计量
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    error_abs_mean = np.mean(np.abs(errors))
    
    print("\n误差分析:")
    print(f"误差均值: {error_mean:.4f}")
    print(f"误差标准差: {error_std:.4f}")
    print(f"绝对误差均值: {error_abs_mean:.4f}")
    
    # 检测并报告异常值(超过3个标准差)
    outlier_indices = np.where(np.abs(errors) > 3 * error_std)[0]
    outlier_percentage = len(outlier_indices) / len(errors) * 100
    print(f"异常值比例 (超过3个标准差): {outlier_percentage:.2f}%")
    
    # 模型偏差分析 - 检查预测是否有系统性偏差
    correlation = np.corrcoef(predictions.flatten(), errors.flatten())[0, 1]
    print(f"预测值与误差的相关性: {correlation:.4f}")
    
    if abs(correlation) > 0.3:
        print("存在显著的模型偏差: 预测值与误差相关，表明模型可能缺少一些关键特征或存在结构性问题。")
    
    # 绘制预测值与误差的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('预测值')
    plt.ylabel('误差')
    plt.title('预测值 vs 误差 (用于检测模型偏差)')
    plt.savefig('d:/amath/B/B1/results/预测值_误差图.png')
    plt.close()
    
    # 误差大小与目标值的关系分析
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, np.abs(errors), alpha=0.5)
    plt.xlabel('真实值')
    plt.ylabel('绝对误差')
    plt.title('真实值 vs 绝对误差 (用于检测数据噪声)')
    plt.savefig('d:/amath/B/B1/results/真实值_绝对误差图.png')
    plt.close()
    
    # 根据数据分析报告可能的误差来源
    print("\n可能的误差来源:")
    if error_mean > 0.1 * np.mean(targets):
        print("- 系统性偏差: 模型预测值与实际值存在系统性差异。")
    
    if outlier_percentage > 5:
        print("- 异常值影响: 数据中存在较多异常值，可能影响模型性能。")
    
    if error_std > 0.3 * np.std(targets):
        print("- 数据噪声: 误差变异较大，可能存在较高的数据噪声。")
    
    if abs(correlation) > 0.3:
        print("- 模型欠拟合: 模型可能缺少一些重要特征或结构复杂度不足。")
