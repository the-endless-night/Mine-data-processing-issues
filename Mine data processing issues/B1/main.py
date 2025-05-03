import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from data_preprocessing import load_data, clean_data, preprocess_data
from model import create_model
from train import train_model
from evaluate import evaluate_model

def main():
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 检查是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建时间戳用于文件命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 文件路径
    a_path = "A.xlsx"
    b_path = "B.xlsx"
    
    # 确保文件存在
    if not os.path.exists(a_path) or not os.path.exists(b_path):
        print(f"错误: 请确保 {a_path} 和 {b_path} 文件在当前目录中!")
        return
    
    # 创建保存结果的目录
    os.makedirs("d:/amath/B/B1/results", exist_ok=True)
    
    print("开始数据加载...")
    df_a, df_b = load_data(a_path, b_path)
    
    print("开始数据清洗与异常值处理...")
    df_a, df_b, outlier_indices = clean_data(df_a, df_b, handle_outliers=True, outlier_method='robust')
    
    print(f"检测到的异常点数量: {len(outlier_indices)}")
    print(f"异常点比例: {len(outlier_indices)/len(df_a)*100:.2f}%")
    
    # 保存异常点信息
    outlier_df = pd.DataFrame({'outlier_index': outlier_indices})
    outlier_df.to_csv('d:/amath/B/B1/results/outliers.csv', index=False)
    
    print("开始数据预处理...")
    train_loader, test_loader, scaler, feature_info = preprocess_data(df_a, df_b)
    
    print("创建并训练FTTransformer模型...")
    model = create_model(device, input_dim=df_a.shape[1], output_dim=df_b.shape[1])
    print(f"模型结构: {model}")
    
    model, train_losses, val_losses = train_model(model, train_loader, test_loader, device)
    
    # 保存模型
    torch.save(model.state_dict(), "d:/amath/B/B1/model.pth")
    
    # 分析异常点影响
    if len(outlier_indices) > 0:
        # 创建不含异常点的测试数据加载器
        clean_train_loader, clean_test_loader, _, _ = preprocess_data(
            df_a[~df_a.index.isin(outlier_indices)], 
            df_b[~df_b.index.isin(outlier_indices)]
        )
        
        # 比较含异常点和不含异常点的模型性能
        print("评估含异常点的模型性能...")
        predictions, targets, mse, rmse, mae, r2 = evaluate_model(model, test_loader, device)
        
        print("评估在无异常点数据上的模型性能...")
        clean_predictions, clean_targets, clean_mse, clean_rmse, clean_mae, clean_r2 = evaluate_model(model, clean_test_loader, device)
        
        # 记录比较结果
        comparison_results = {
            "含异常点_MSE": mse,
            "含异常点_RMSE": rmse,
            "含异常点_MAE": mae,
            "含异常点_R2": r2,
            "无异常点_MSE": clean_mse,
            "无异常点_RMSE": clean_rmse,
            "无异常点_MAE": clean_mae,
            "无异常点_R2": clean_r2,
            "性能提升_MSE": (mse - clean_mse) / mse * 100 if mse > 0 else 0,
            "性能提升_RMSE": (rmse - clean_rmse) / rmse * 100 if rmse > 0 else 0
        }
        
        # 保存比较结果
        comp_df = pd.DataFrame([comparison_results])
        comp_df.to_csv(f"d:/amath/B/B1/results/异常点影响分析_{timestamp}.csv", index=False)
        print("异常点影响分析完成!")
    else:
        print("评估模型性能...")
        predictions, targets, mse, rmse, mae, r2 = evaluate_model(model, test_loader, device)
    
    # 将评估结果保存到文件
    results = {
        "MSE": mse,
        "RMSE": rmse, 
        "MAE": mae,
        "R2": r2,
        "异常点数量": len(outlier_indices),
        "异常点比例": len(outlier_indices)/len(df_a)*100,
        "Timestamp": timestamp
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"d:/amath/B/B1/results/evaluation_{timestamp}.csv", index=False)
    
    print("分析完成!")
    print(f"结果已保存到 d:/amath/B/B1/results/ 目录")

if __name__ == "__main__":
    main()
