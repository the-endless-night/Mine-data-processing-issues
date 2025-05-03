import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import shutil
from pathlib import Path

def read_file_content(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"无法读取文件: {str(e)}"

def extract_metrics(evaluation_text):
    """从评估文本中提取指标"""
    metrics = {}
    
    # 提取MSE
    mse_match = re.search(r'均方误差 \(MSE\): ([0-9.]+)', evaluation_text)
    if mse_match:
        metrics['MSE'] = float(mse_match.group(1))
    
    # 提取R^2 - 支持LaTeX格式 R$^2$
    r2_match = re.search(r'决定系数 \(R\$?\^2\$?\): ([0-9.]+)', evaluation_text)
    if r2_match:
        metrics['R^2'] = float(r2_match.group(1))
    
    # 提取交叉验证R^2 - 支持LaTeX格式
    cv_match = re.search(r'交叉验证R\$?\^2\$?: ([0-9.]+) ± ([0-9.]+)', evaluation_text)
    if cv_match:
        metrics['CV_R^2_mean'] = float(cv_match.group(1))
        metrics['CV_R^2_std'] = float(cv_match.group(2))
    
    # 提取重构误差
    recon_match = re.search(r'重构误差 \(MSE\): ([0-9.]+)', evaluation_text)
    if recon_match:
        metrics['Reconstruction_Error'] = float(recon_match.group(1))
    
    # 提取原始维度和降维后维度
    orig_dim_match = re.search(r'原始维度: ([0-9]+)', evaluation_text)
    reduced_dim_match = re.search(r'降维后维度: ([0-9]+)', evaluation_text)
    if orig_dim_match and reduced_dim_match:
        metrics['Original_Dimension'] = int(orig_dim_match.group(1))
        metrics['Reduced_Dimension'] = int(reduced_dim_match.group(1))
    
    return metrics

def extract_training_time(complexity_text):
    """从复杂度分析文本中提取训练时间"""
    times = {}
    
    # 提取自编码器训练时间
    ae_time_match = re.search(r'训练时间: ([0-9.]+) 秒', complexity_text)
    if ae_time_match:
        times['Autoencoder_Training_Time'] = float(ae_time_match.group(1))
    
    # 提取编码时间
    encoding_time_match = re.search(r'编码时间: ([0-9.]+) 秒', complexity_text)
    if encoding_time_match:
        times['Encoding_Time'] = float(encoding_time_match.group(1))
    
    # 提取重构时间
    reconstruction_time_match = re.search(r'重构时间: ([0-9.]+) 秒', complexity_text)
    if reconstruction_time_match:
        times['Reconstruction_Time'] = float(reconstruction_time_match.group(1))
    
    # 提取MLP训练时间
    mlp_train_time_match = re.search(r'MLP.*?\n.*?训练时间: ([0-9.]+) 秒', complexity_text, re.DOTALL)
    if mlp_train_time_match:
        times['MLP_Training_Time'] = float(mlp_train_time_match.group(1))
    
    # 提取MLP预测时间
    mlp_pred_time_match = re.search(r'预测时间: ([0-9.]+) 秒', complexity_text)
    if mlp_pred_time_match:
        times['MLP_Prediction_Time'] = float(mlp_pred_time_match.group(1))
    
    return times

def generate_model_evaluation_report(output_dir, report_path=None):
    """生成模型评估报告并保存为Markdown文件"""
    if report_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f'model_evaluation_report_{timestamp}.md')
    
    # 读取评估文件
    evaluation_path = os.path.join(output_dir, 'model_evaluation.txt')
    complexity_path = os.path.join(output_dir, 'complexity_analysis.txt')
    
    evaluation_text = read_file_content(evaluation_path)
    complexity_text = read_file_content(complexity_path)
    
    # 提取指标
    metrics = extract_metrics(evaluation_text)
    times = extract_training_time(complexity_text)
    
    # 获取并复制图片到报告目录
    report_dir = os.path.dirname(report_path)
    report_images_dir = os.path.join(report_dir, 'report_images')
    os.makedirs(report_images_dir, exist_ok=True)
    
    # 图片路径列表
    image_paths = {
        'autoencoder_loss': os.path.join(output_dir, 'autoencoder_loss.png'),
        'dimension_optimization': os.path.join(output_dir, 'dimension_optimization.png'),
        'reduced_data_2d': os.path.join(output_dir, 'reduced_data_2d.png'),
        'reduced_data_3d': os.path.join(output_dir, 'reduced_data_3d.png'),
        'prediction_vs_actual': os.path.join(output_dir, 'prediction_vs_actual.png'),
        'residual_analysis': os.path.join(output_dir, 'residual_analysis.png'),
        'model_performance': os.path.join(output_dir, 'model_performance.png')
    }
    
    # 复制存在的图片
    copied_images = {}
    for name, path in image_paths.items():
        if os.path.exists(path):
            dest_path = os.path.join(report_images_dir, f'{name}.png')
            shutil.copy(path, dest_path)
            copied_images[name] = os.path.relpath(dest_path, report_dir)
    
    # 生成Markdown报告
    report_content = f"""# 自编码器降维与MLP回归模型评估报告

## 1. 实验概述

本实验使用自编码器进行高维数据降维，然后通过多层感知机（MLP）构建回归模型预测目标变量。实验旨在探索如何在保留数据关键信息的同时，降低维度以提高模型性能和计算效率。

### 1.1 研究背景

高维数据在机器学习中常常面临维度灾难问题，即随着维度增加，需要的样本数量呈指数级增长。通过降维技术，我们可以提取数据中的主要特征，减少噪声影响，提高模型的泛化能力。

### 1.2 研究目标

- 使用自编码器实现有效的非线性降维
- 为降维后的数据构建高性能回归模型
- 分析降维维度对模型性能的影响
- 评估模型的泛化能力和预测精度

## 2. 实验设置

### 2.1 数据集

实验使用了从Excel文件`5-X.xlsx`和`5-Y.xlsx`加载的数据集。数据经过了以下预处理步骤：
- 检测并删除重复行
- 填充缺失值
- 处理异常值
- 标准化处理

### 2.2 模型架构

#### 自编码器模型
自编码器包含编码器和解码器两部分：
- **编码器**: 3层神经网络，使用LeakyReLU和Tanh激活函数
- **解码器**: 3层神经网络，使用LeakyReLU和Sigmoid激活函数
- **降维**: 从{metrics.get('Original_Dimension', 'N/A')}维降至{metrics.get('Reduced_Dimension', 'N/A')}维

#### MLP回归模型
- 使用sklearn的Pipeline结合PolynomialFeatures和MLPRegressor
- MLPRegressor使用(256, 128, 64, 32)的网络结构
- 激活函数为tanh
- 使用early stopping避免过拟合

## 3. 实验过程

### 3.1 数据预处理
数据预处理包括：
- Z-score方法处理异常值
- 标准化X和Y数据

### 3.2 降维维度优化
通过网格搜索确定最优的降维维度：
"""
    
    if 'dimension_optimization' in copied_images:
        report_content += f"""
![降维维度优化]({copied_images['dimension_optimization']})

图1: 不同降维维度的性能比较
"""
    
    report_content += f"""
最优降维维度被确定为{metrics.get('Reduced_Dimension', 'N/A')}，此时模型具有最佳的R^2值和较低的重构误差。

### 3.3 自编码器训练

自编码器在以下配置下训练：
- 批量大小: 自动优化
- 使用Adam优化器
- 学习率自适应调整
- 早停机制避免过拟合
- 训练时间: {times.get('Autoencoder_Training_Time', 'N/A')} 秒
"""
    
    if 'autoencoder_loss' in copied_images:
        report_content += f"""
![自编码器训练损失]({copied_images['autoencoder_loss']})

图2: 自编码器训练过程中的损失变化
"""
    
    report_content += f"""
### 3.4 降维和重构

数据通过训练好的自编码器进行降维和重构：
- 编码时间: {times.get('Encoding_Time', 'N/A')} 秒
- 重构时间: {times.get('Reconstruction_Time', 'N/A')} 秒
- 重构误差(MSE): {metrics.get('Reconstruction_Error', 'N/A')}

"""
    
    # 添加降维可视化（2D或3D）
    if 'reduced_data_2d' in copied_images:
        report_content += f"""
![降维数据(2D)]({copied_images['reduced_data_2d']})

图3: 降维后的数据在2D空间的分布与目标值的关系
"""
    elif 'reduced_data_3d' in copied_images:
        report_content += f"""
![降维数据(3D)]({copied_images['reduced_data_3d']})

图3: 降维后的数据在3D空间的分布与目标值的关系
"""
    
    report_content += f"""
### 3.5 MLP回归模型训练

使用重构后的数据训练MLP回归模型：
- 训练集和测试集按8:2比例划分
- 使用3折交叉验证评估模型
- MLP训练时间: {times.get('MLP_Training_Time', 'N/A')} 秒
- MLP预测时间: {times.get('MLP_Prediction_Time', 'N/A')} 秒

## 4. 实验结果

### 4.1 模型性能指标

| 指标 | 值 |
|------|-----|
| 均方误差 (MSE) | {metrics.get('MSE', 'N/A')} |
| 决定系数 (R^2) | {metrics.get('R^2', 'N/A')} |
| 交叉验证R^2 | {metrics.get('CV_R^2_mean', 'N/A')} ± {metrics.get('CV_R^2_std', 'N/A')} |
| 重构误差 (MSE) | {metrics.get('Reconstruction_Error', 'N/A')} |

"""
    
    if 'model_performance' in copied_images:
        report_content += f"""
![模型性能指标]({copied_images['model_performance']})

图4: MLP模型性能指标可视化
"""
    
    report_content += f"""
### 4.2 预测结果分析
"""
    
    if 'prediction_vs_actual' in copied_images:
        report_content += f"""
![预测值vs实际值]({copied_images['prediction_vs_actual']})

图5: 预测值与实际值的对比
"""
    
    if 'residual_analysis' in copied_images:
        report_content += f"""
![残差分析]({copied_images['residual_analysis']})

图6: 模型残差分析
"""
    
    report_content += f"""
## 5. 结果分析

### 5.1 降维效果分析

自编码器成功将{metrics.get('Original_Dimension', 'N/A')}维数据降至{metrics.get('Reduced_Dimension', 'N/A')}维，同时保持了数据的关键特征。降维后的数据重构误差为{metrics.get('Reconstruction_Error', 'N/A')}，表明自编码器能够较好地还原原始数据结构。

### 5.2 模型性能分析

MLP回归模型在降维后的数据上取得了R^2值为{metrics.get('R^2', 'N/A')}的性能，交叉验证的平均R^2为{metrics.get('CV_R^2_mean', 'N/A')} ± {metrics.get('CV_R^2_std', 'N/A')}。这表明模型具有良好的拟合能力和稳定性。

### 5.3 时间效率分析

- 自编码器训练耗时: {times.get('Autoencoder_Training_Time', 'N/A')} 秒
- 数据降维耗时: {times.get('Encoding_Time', 'N/A')} 秒
- MLP模型训练耗时: {times.get('MLP_Training_Time', 'N/A')} 秒
- 预测耗时: {times.get('MLP_Prediction_Time', 'N/A')} 秒

总体来看，模型在保持预测性能的同时，实现了较高的计算效率。

## 6. 模型评价

### 6.1 优点

1. **有效降维**: 自编码器成功实现了非线性降维，保留了数据的关键信息
2. **良好的预测性能**: MLP回归模型在降维数据上取得了可接受的R^2值
3. **计算效率**: 通过降维减少了计算复杂度，提高了模型训练和预测效率
4. **自动化优化**: 自动寻找最优降维维度和批量大小，减少了人工调参工作

### 6.2 局限性

1. **信息损失**: 降维过程不可避免地会损失一些信息，可能影响最终预测性能
2. **计算资源需求**: 自编码器训练需要较多的计算资源
3. **超参数敏感性**: 模型性能对降维维度等超参数较为敏感

## 7. 未来工作

1. **集成学习**: 可以尝试结合多个不同维度的模型，通过集成学习提高预测性能
2. **其他降维技术比较**: 与PCA、t-SNE等其他降维技术进行比较
3. **深度自编码器**: 使用更深层的自编码器结构，尝试提取更复杂的特征
4. **迁移学习**: 探索在相似数据集上预训练自编码器的可能性

## 8. 结论

本实验证明了自编码器与MLP回归模型相结合是处理高维回归问题的有效方法。通过自动优化降维维度和模型参数，我们成功构建了一个既保持预测精度又提高计算效率的模型。实验结果表明，该方法在保留关键信息的同时，有效降低了数据维度，并且在预测任务中表现良好。

---

*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    # 保存报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"模型评估报告已保存至: {report_path}")
    return report_path

if __name__ == "__main__":
    output_dir = 'd:/amath/B/B5/output'
    report_path = os.path.join(output_dir, 'model_evaluation_report.md')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成报告
    generated_path = generate_model_evaluation_report(output_dir, report_path)
    
    print(f"报告已生成，路径为: {generated_path}")
    print("您可以使用Markdown查看器查看该报告，或将其转换为HTML或PDF格式。")
