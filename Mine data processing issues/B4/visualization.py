import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.metrics import mean_squared_error

# 设置matplotlib参数，确保特殊字符正确显示
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']  # 支持中文和特殊字符
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def visualize_results(results, X_test, y_test):
    """可视化不同模型的预测结果"""
    plt.figure(figsize=(15, 10))
    
    # 获取所有模型和它们的R²分数
    models = list(results.keys())
    r2_scores = [results[model]["metrics"]["r2"] for model in models]
    
    # 按R²分数排序
    sorted_indices = np.argsort(r2_scores)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_scores = [r2_scores[i] for i in sorted_indices]
    
    # 绘制模型比较图
    plt.subplot(2, 2, 1)
    plt.bar(sorted_models, sorted_scores)
    plt.xlabel('模型')
    plt.ylabel('$R^2$ 分数')  # 使用LaTeX格式确保正确显示
    plt.title('不同模型的$R^2$分数比较')
    plt.xticks(rotation=45)
    
    # 绘制最佳模型的预测vs实际值图
    best_model_name = sorted_models[0]
    best_model = results[best_model_name]["model"]
    y_pred = best_model.predict(X_test)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'最佳模型 ({best_model_name}) 预测vs实际值')
    
    # 绘制误差分布图
    errors = y_test - y_pred
    plt.subplot(2, 2, 3)
    plt.hist(errors, bins=20)
    plt.xlabel('预测误差')
    plt.ylabel('频率')
    plt.title('预测误差分布')
    
    # 绘制模型稳定性对比
    stabilities = [results[model]["stability"] for model in sorted_models]
    
    plt.subplot(2, 2, 4)
    plt.bar(sorted_models, stabilities)
    plt.xlabel('模型')
    plt.ylabel('稳定性指标')
    plt.title('不同模型的稳定性比较')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def plot_parameter_correlation(params_history, scores_history, model_name):
    """可视化参数与拟合优度的相关性"""
    if not params_history or not scores_history:
        return
    
    param_keys = list(params_history[0].keys())
    n_params = len(param_keys)
    
    if n_params == 0:
        return
    
    # 为每个参数创建一个图
    plt.figure(figsize=(15, 5 * ((n_params + 1) // 2)))
    
    for i, param_key in enumerate(param_keys):
        plt.subplot((n_params + 1) // 2, 2, i + 1)
        
        # 提取参数值
        param_values = []
        param_labels = []
        
        for params in params_history:
            value = params[param_key]
            # 确保值是数值型或可以转换为数值索引
            if isinstance(value, (int, float)):
                param_values.append(value)
                param_labels.append(str(value))
            elif isinstance(value, tuple):
                # 对于神经网络的隐藏层大小，使用第一个值
                param_values.append(value[0])
                param_labels.append(str(value))
            else:
                # 对于分类参数，转为数值索引
                unique_values = sorted(set(p[param_key] for p in params_history if param_key in p))
                param_values.append(unique_values.index(value))
                param_labels.append(str(value))
        
        if len(param_values) == len(scores_history):
            # 使用颜色映射显示参数-性能关系
            scatter = plt.scatter(param_values, scores_history, 
                                 c=scores_history, cmap='viridis', 
                                 alpha=0.7)
            plt.colorbar(scatter, label='$R^2$ 分数')  # 使用LaTeX格式
            
            # 如果参数是分类变量，调整x轴标签
            if isinstance(params_history[0][param_key], (str, tuple)):
                unique_values = sorted(set(p[param_key] for p in params_history))
                unique_indices = range(len(unique_values))
                plt.xticks(unique_indices, [str(v) for v in unique_values], rotation=45)
            
            # 添加趋势线
            if len(param_values) > 1 and all(isinstance(v, (int, float)) for v in param_values):
                try:
                    z = np.polyfit(param_values, scores_history, 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(param_values), p(sorted(param_values)), "r--")
                    
                    # 计算相关系数
                    correlation = np.corrcoef(param_values, scores_history)[0, 1]
                    plt.title(f'{param_key} 与拟合优度的关系 (相关性: {correlation:.3f})')
                except:
                    plt.title(f'{param_key} 与拟合优度的关系')
            else:
                plt.title(f'{param_key} 与拟合优度的关系')
                
            plt.xlabel(param_key)
            plt.ylabel('$R^2$ 分数')  # 使用LaTeX格式
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_parameter_correlation.png')
    plt.close()

def visualize_mlp_results(model, X_test, y_test, metrics, model_name="mlp"):
    """可视化多层感知机模型的预测结果"""
    plt.figure(figsize=(15, 12))
    
    # 预测值
    y_pred = model.predict(X_test)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # 绘制预测vs实际值图
    plt.subplot(3, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'{model_name} - 预测vs实际值 ($R^2$ = {metrics["r2"]:.4f})')  # 使用LaTeX格式
    
    # 绘制误差分布图
    errors = y_test - y_pred
    plt.subplot(3, 2, 2)
    plt.hist(errors, bins=20)
    plt.xlabel('预测误差')
    plt.ylabel('频率')
    plt.title(f'预测误差分布 (RMSE = {metrics["rmse"]:.4f})')
    
    # 绘制残差图
    plt.subplot(3, 2, 3)
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差分布图')
    
    # 绘制Q-Q图
    from scipy import stats
    plt.subplot(3, 2, 4)
    stats.probplot(errors.flatten(), dist="norm", plot=plt)
    plt.title('Q-Q图 - 误差正态性检验')
    
    # 绘制预测值与实际值的时间序列对比
    plt.subplot(3, 2, 5)
    plt.plot(y_test, 'b-', label='实际值')
    plt.plot(y_pred, 'r-', label='预测值')
    plt.legend()
    plt.xlabel('样本索引')
    plt.ylabel('值')
    plt.title('预测值与实际值对比')
    
    # 绘制相对误差直方图
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = np.abs(y_test - y_pred) / np.abs(y_test) * 100
        relative_errors = np.clip(relative_errors, 0, 100)  # 限制范围，便于可视化
    
    plt.subplot(3, 2, 6)
    plt.hist(relative_errors, bins=20)
    plt.xlabel('相对误差 (%)')
    plt.ylabel('频率')
    plt.title(f'相对误差分布 (平均: {metrics["mre"]:.2f}%)')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_results.png')
    plt.close()

def plot_optimization_convergence(convergence_history, model_name):
    """绘制优化收敛过程"""
    if not convergence_history:
        return
    
    plt.figure(figsize=(10, 6))
    
    iterations = [item[0] for item in convergence_history]
    scores = [item[1] for item in convergence_history]
    
    plt.plot(iterations, scores, 'o-')
    plt.axhline(y=max(scores), color='r', linestyle='--', alpha=0.5, 
                label=f'最佳分数: {max(scores):.4f}')
    
    # 标记最佳点
    best_idx = scores.index(max(scores))
    plt.scatter([iterations[best_idx]], [scores[best_idx]], color='red', s=100, 
                label=f'最佳点: 迭代{iterations[best_idx]}')
    
    plt.xlabel('迭代次数')
    plt.ylabel('$R^2$ 分数')  # 使用LaTeX格式
    plt.title(f'{model_name} - 优化收敛过程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_convergence.png')
    plt.close()

def plot_feature_importance(model, model_name):
    """绘制特征重要性（仅适用于支持feature_importance_属性的模型）"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [f'特征 {i}' for i in indices], rotation=90)
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.title(f'{model_name} - 特征重要性')
        plt.tight_layout()
        plt.savefig(f'{model_name}_feature_importance.png')
        plt.close()

def visualize_error_analysis(model, X_test, y_test, model_name):
    """可视化误差分析"""
    y_pred = model.predict(X_test)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    errors = y_test - y_pred
    abs_errors = np.abs(errors)
    
    # 找出误差最大的样本
    n_worst = min(10, len(X_test))
    worst_indices = np.argsort(abs_errors.flatten())[-n_worst:]
    
    plt.figure(figsize=(12, 8))
    
    # 绘制误差分布热图
    plt.subplot(2, 1, 1)
    error_data = pd.DataFrame({
        '预测值': y_pred.flatten(),
        '实际值': y_test.flatten(),
        '绝对误差': abs_errors.flatten()
    })
    
    # 按误差大小排序
    error_data = error_data.sort_values('绝对误差', ascending=False)
    
    # 创建热图数据，按预测值和实际值分区
    pred_bins = np.linspace(y_pred.min(), y_pred.max(), 10)
    actual_bins = np.linspace(y_test.min(), y_test.max(), 10)
    
    # 计算每个区间的平均误差
    heatmap_data = np.zeros((len(pred_bins)-1, len(actual_bins)-1))
    for i in range(len(pred_bins)-1):
        for j in range(len(actual_bins)-1):
            mask = ((y_pred >= pred_bins[i]) & (y_pred < pred_bins[i+1]) & 
                    (y_test >= actual_bins[j]) & (y_test < actual_bins[j+1]))
            if np.any(mask):
                heatmap_data[i, j] = np.mean(abs_errors[mask])
    
    sns.heatmap(heatmap_data, cmap='YlOrRd', 
               xticklabels=[f'{v:.2f}' for v in actual_bins],
               yticklabels=[f'{v:.2f}' for v in pred_bins])
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('误差热图分析')
    
    # 绘制误差最大的样本对比
    plt.subplot(2, 1, 2)
    width = 0.35
    x = np.arange(len(worst_indices))
    
    plt.bar(x - width/2, y_test[worst_indices].flatten(), width, label='实际值')
    plt.bar(x + width/2, y_pred[worst_indices].flatten(), width, label='预测值')
    
    plt.xlabel('样本索引')
    plt.ylabel('值')
    plt.title('误差最大的样本对比')
    plt.legend()
    plt.xticks(x, [f'{i}' for i in worst_indices])
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_error_analysis.png')
    plt.close()

def plot_model_comparison(all_metrics, all_stability):
    """比较多个模型的性能"""
    if not all_metrics:
        return
    
    model_names = list(all_metrics.keys())
    metrics_of_interest = ['r2', 'rmse', 'mae', 'mre']
    metrics_titles = {
        'r2': '$R^2$ 分数 (越高越好)',  # 使用LaTeX格式
        'rmse': '均方根误差 (越低越好)',
        'mae': '平均绝对误差 (越低越好)',
        'mre': '平均相对误差 % (越低越好)'
    }
    
    plt.figure(figsize=(15, 10))
    
    # 绘制各指标对比图
    for i, metric in enumerate(metrics_of_interest):
        plt.subplot(2, 2, i+1)
        
        values = [all_metrics[model][metric] for model in model_names]
        
        if metric == 'r2':  # R²越高越好
            colors = ['green' if v > 0.8 else 'orange' if v > 0.5 else 'red' for v in values]
        else:  # 其他指标越低越好
            max_val = max(values)
            colors = ['green' if v < max_val/3 else 'orange' if v < 2*max_val/3 else 'red' for v in values]
        
        bars = plt.bar(model_names, values, color=colors)
        plt.xlabel('模型')
        plt.ylabel(metrics_titles[metric] if metric in metrics_titles else metric)
        plt.title(metrics_titles[metric])
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_metrics_comparison.png')
    plt.close()
    
    # 绘制稳定性和计算时间对比
    plt.figure(figsize=(10, 6))
    stability_values = list(all_stability.values())
    
    plt.bar(model_names, stability_values)
    plt.xlabel('模型')
    plt.ylabel('稳定性指标')
    plt.title('模型稳定性比较 (越高越稳定)')
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for i, v in enumerate(stability_values):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_stability_comparison.png')
    plt.close()
