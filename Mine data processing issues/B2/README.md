# 矿山监测数据压缩与还原系统

本系统使用PyTorch实现了基于自动编码器的矿山监测数据压缩和还原模型，可以有效降低数据存储需求，同时保证还原数据的准确性。

## 功能特点

- 使用自动编码器实现数据降维和还原
- 支持GPU加速训练和推理
- 可配置的编码维度和网络结构
- 自动计算压缩效率指标（压缩比、存储空间节省率等）
- 还原数据的准确度评估（MSE、MAE等）
- 可视化训练过程和数据对比

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- NumPy
- Pandas
- Matplotlib
- scikit-learn

可以通过以下命令安装依赖：

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

## 使用方法

### 1. 训练模型

```bash
python mine_data_compression.py --data_file 数据文件路径.csv --output_dir 输出目录
```

可选参数：
- `--batch_size`: 批次大小（默认：64）
- `--epochs`: 训练轮数（默认：200）
- `--learning_rate`: 学习率（默认：0.001）
- `--encoding_dim`: 编码维度，0表示根据压缩率自动计算（默认：0）
- `--compression_ratio`: 当编码维度为0时使用的压缩率（默认：4.0）
- `--hidden_dims`: 隐藏层维度，逗号分隔（默认自动设置）
- `--mse_threshold`: MSE阈值（默认：0.005）

### 2. 测试模型

```bash
python test_model.py --model_file 模型文件路径.pth --data_file 测试数据路径.csv
```

可选参数：
- `--output_dir`: 输出目录（默认：test_output）
- `--batch_size`: 批次大小（默认：64）
- `--mse_threshold`: MSE阈值（默认：0.005）

## 输出说明

训练模型后，系统会在输出目录生成以下文件：

1. `autoencoder_model.pth`: 训练好的模型文件
2. `training_history.png`: 训练过程中损失变化的图表
3. `data_comparison.png`: 原始数据和重构数据的对比图

测试模型后，系统会在输出目录生成：

1. `test_data_comparison.png`: 测试数据和重构数据的对比图
2. `samples.npz`: 部分测试样本和对应的重构结果

## 性能调优

为了在保证还原数据准确度的前提下提高压缩效率，可以尝试：

1. 调整编码维度（`--encoding_dim`）：较小的编码维度会提高压缩率，但可能降低还原准确度
2. 调整网络结构（`--hidden_dims`）：更复杂的网络结构可能提高还原准确度
3. 增加训练轮数（`--epochs`）：更充分的训练可能提高模型性能
4. 调整学习率和正则化参数：防止过拟合，提高泛化能力

## 注意事项

- 确保输入数据是结构化的矿山监测数据，支持CSV和Excel格式
- 系统会自动选择数值型列进行处理，忽略非数值型列
- 训练数据会自动进行标准化处理
- 在评估时，系统会将数据转换回原始尺度进行分析
