import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_data_loaders
from model import Autoencoder
from utils import train_model, evaluate_model, plot_training_history, plot_data_comparison, analyze_compression_efficiency

def main():
    # 硬编码参数
    class Args:
        def __init__(self):
            self.data_file = r"D:\amath\B\B2\Data.xlsx"  # 使用原始字符串表示法(r前缀)解决路径问题
            self.output_dir = "results"
            self.batch_size = 64
            self.epochs = 200
            self.learning_rate = 0.001
            self.weight_decay = 1e-5
            self.encoding_dim = 0  # 0表示根据压缩率自动计算
            self.compression_ratio = 4.0
            self.hidden_dims = ""  # 空字符串表示自动创建隐藏层结构
            self.test_split = 0.2
            self.patience = 20
            self.mse_threshold = 0.005
            self.no_cuda = False
            self.continue_training = False
    
    args = Args()
    
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载数据
    print(f"正在加载数据: {args.data_file}")
    train_loader, test_loader, dataset = get_data_loaders(
        args.data_file, 
        batch_size=args.batch_size,
        test_split=args.test_split
    )
    
    input_dim = dataset.get_feature_dim()
    print(f"数据特征维度: {input_dim}")
    
    # 创建模型
    encoding_dim = args.encoding_dim
    if encoding_dim <= 0:  # 如果没有指定编码维度，则根据压缩率计算
        encoding_dim = max(1, int(input_dim / args.compression_ratio))
    
    if args.hidden_dims:
        hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    else:
        # 自动创建一个合理的隐藏层结构
        hidden_dims = [input_dim * 2, input_dim]
    
    print(f"创建自动编码器模型: 输入维度={input_dim}, 编码维度={encoding_dim}, 隐藏层={hidden_dims}")
    model = Autoencoder(input_dim, encoding_dim, hidden_dims)
    
    # 打印模型结构
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 保存模型路径
    model_path = os.path.join(args.output_dir, 'autoencoder_model.pth')
    
    # 检查是否已有训练好的模型
    if os.path.exists(model_path):
        print(f"发现已训练好的模型: {model_path}")
        print("正在加载模型...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 确保模型结构与已保存的模型匹配
        input_dim = checkpoint['input_dim']
        encoding_dim = checkpoint['encoding_dim']
        hidden_dims = checkpoint['hidden_dims']
        
        # 创建相同结构的模型
        model = Autoencoder(input_dim, encoding_dim, hidden_dims)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # 直接使用已保存的评估指标
        if checkpoint['metrics'] is not None:
            print("使用已保存的评估指标...")
            metrics = checkpoint['metrics']
            
            # 直接分析压缩效率
            compression_analysis = analyze_compression_efficiency(model, metrics)
            
            # 检查MSE是否满足要求
            if metrics['mse'] <= args.mse_threshold:
                print(f"模型满足MSE要求: {metrics['mse']:.6f} <= {args.mse_threshold}")
            else:
                print(f"警告: 模型MSE ({metrics['mse']:.6f}) 高于阈值 {args.mse_threshold}")
            
            # 绘制数据对比
            comparison_path = os.path.join(args.output_dir, 'data_comparison.png')
            plot_data_comparison(
                metrics['original_data'],
                metrics['reconstructed_data'],
                feature_indices=range(min(5, input_dim)),
                num_samples=20,
                save_path=comparison_path
            )
            
            print(f"已使用训练好的模型: {model_path}")
            print("完成!")
            return  # 直接结束程序
        else:
            print("模型已加载，但未找到保存的评估指标，将进行评估...")
            
        print("模型加载完成!")
    else:
        print("未找到已训练的模型，开始训练新模型...")
        # 训练模型
        print(f"开始训练模型...")
        model, history = train_model(
            model, 
            train_loader, 
            test_loader,
            criterion, 
            optimizer,
            device,
            num_epochs=args.epochs,
            patience=args.patience
        )
        
        # 绘制训练历史
        history_path = os.path.join(args.output_dir, 'training_history.png')
        plot_training_history(history, save_path=history_path)
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'input_dim': input_dim,
            'encoding_dim': encoding_dim,
            'hidden_dims': hidden_dims,
            'metrics': None  # 初始时没有评估指标
        }, model_path)
        print(f"新模型保存至: {model_path}")
    
    # 评估模型性能（无论是新训练的还是加载的）
    print("评估模型性能...")
    metrics = evaluate_model(model, test_loader, dataset, device)
    
    # 更新保存的模型以包含最新的评估指标
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint['metrics'] = metrics
        torch.save(checkpoint, model_path)
    
    # 分析压缩效率
    compression_analysis = analyze_compression_efficiency(model, metrics)
    
    # 检查MSE是否满足要求
    if metrics['mse'] <= args.mse_threshold:
        print(f"模型满足MSE要求: {metrics['mse']:.6f} <= {args.mse_threshold}")
    else:
        print(f"警告: 模型MSE ({metrics['mse']:.6f}) 高于阈值 {args.mse_threshold}")
    
    # 绘制数据对比
    comparison_path = os.path.join(args.output_dir, 'data_comparison.png')
    plot_data_comparison(
        metrics['original_data'],
        metrics['reconstructed_data'],
        feature_indices=range(min(5, input_dim)),
        num_samples=20,
        save_path=comparison_path
    )
    
    print(f"模型保存至: {model_path}")
    print("完成!")

if __name__ == "__main__":
    main()
