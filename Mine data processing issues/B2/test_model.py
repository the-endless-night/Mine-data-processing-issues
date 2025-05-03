import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import MineDataset
from model import Autoencoder
from utils import evaluate_model, analyze_compression_efficiency, plot_data_comparison

def load_model(model_path, device):
    """加载已保存的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    input_dim = checkpoint['input_dim']
    encoding_dim = checkpoint['encoding_dim']
    hidden_dims = checkpoint['hidden_dims']
    
    model = Autoencoder(input_dim, encoding_dim, hidden_dims)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint

def main():
    # 硬编码参数
    class Args:
        def __init__(self):
            self.model_file = r"results\autoencoder_model.pth"
            self.data_file = r"D:\amath\B\B2\Data.xlsx"  # 使用原始字符串表示法避免路径问题
            self.output_dir = "test_results"
            self.batch_size = 64
            self.mse_threshold = 0.005
            self.no_cuda = False
    
    args = Args()
    
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载模型
    print(f"正在加载模型: {args.model_file}")
    model, checkpoint = load_model(args.model_file, device)
    
    # 加载测试数据
    print(f"正在加载测试数据: {args.data_file}")
    dataset = MineDataset(args.data_file)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 评估模型
    print("正在评估模型性能...")
    metrics = evaluate_model(model, test_loader, dataset, device)
    
    # 分析压缩效率
    compression_analysis = analyze_compression_efficiency(model, metrics)
    
    # 检查MSE是否满足要求
    if metrics['mse'] <= args.mse_threshold:
        print(f"模型满足MSE要求: {metrics['mse']:.6f} <= {args.mse_threshold}")
    else:
        print(f"警告: 模型MSE ({metrics['mse']:.6f}) 高于阈值 {args.mse_threshold}")
    
    # 绘制数据对比
    comparison_path = os.path.join(args.output_dir, 'test_data_comparison.png')
    plot_data_comparison(
        metrics['original_data'],
        metrics['reconstructed_data'],
        feature_indices=range(min(5, model.input_dim)),
        num_samples=20,
        save_path=comparison_path
    )
    
    # 保存一些数据样本和对应的重构结果，用于详细分析
    sample_path = os.path.join(args.output_dir, 'samples.npz')
    np.savez(
        sample_path,
        original=metrics['original_data_orig_scale'][:100],
        reconstructed=metrics['reconstructed_data_orig_scale'][:100]
    )
    
    print(f"样本数据保存至: {sample_path}")
    print("完成!")

if __name__ == "__main__":
    main()
