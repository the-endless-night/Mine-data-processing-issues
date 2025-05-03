import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from data_loader import load_and_preprocess_data, inverse_transform_y
from model_builder import ModelBuilder
from evaluator import ModelEvaluator, perform_cross_validation
from visualization import (visualize_mlp_results, plot_parameter_correlation, 
                         plot_optimization_convergence, plot_feature_importance,
                         visualize_error_analysis, plot_model_comparison)
import time
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
if __name__ == "__main__":
    # 1. 加载和预处理数据
    X_data, y_data = load_and_preprocess_data("4-X.xlsx", "4-Y.xlsx")
    print(f"数据加载完成，X形状: {X_data.shape}, y形状: {y_data.shape}")
    
    # 2. 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    # 3. 只训练多层感知机模型
    model_types = ['mlp']  # 只使用多层感知机
    optimization_methods = ['bayesian']  # 可选: 'bayesian', 'genetic', 'random'
    
    all_models = {}
    all_metrics = {}
    all_stability = {}
    
    for model_type in model_types:
        for opt_method in optimization_methods:
            print(f"\n===== 训练多层感知机模型 (使用 {opt_method} 优化) =====")
            
            # 构建模型并进行自适应参数调整
            start_time = time.time()
            model_builder = ModelBuilder(model_type=model_type, optimization_method=opt_method)
            
            # 使用Adam优化器，修改参数网格
            model_builder.param_grid["solver"] = ['adam']  # Adam优化器
            print("指定使用Adam优化器")
            
            # 训练模型
            model, params_history, scores_history, convergence_history = model_builder.build_and_optimize(
                X_train, y_train, X_test, y_test, n_iterations=30, early_stopping=5
            )
            
            training_time = time.time() - start_time
            
            # 评估模型
            evaluator = ModelEvaluator(model)
            metrics = evaluator.evaluate(X_test, y_test)
            stability = evaluator.evaluate_stability(X_data, y_data)
            
            model_name = f"mlp_{opt_method}"
            all_models[model_name] = {
                "model": model,
                "params_history": params_history,
                "scores_history": scores_history,
                "convergence_history": convergence_history,
                "training_time": training_time
            }
            all_metrics[model_name] = metrics
            all_stability[model_name] = stability
            
            print(f"多层感知机模型评估结果 (使用 {opt_method} 优化):")
            print(f"  R² 分数: {metrics['r2']:.4f}")
            print(f"  均方误差 (MSE): {metrics['mse']:.4f}")
            print(f"  均方根误差 (RMSE): {metrics['rmse']:.4f}")
            print(f"  平均绝对误差 (MAE): {metrics['mae']:.4f}")
            print(f"  均方对数误差 (MSLE): {metrics['msle']:.4f}")
            print(f"  平均相对误差 (MRE): {metrics['mre']:.4f}%")
            print(f"  稳定性指标: {stability:.4f}")
            print(f"  训练时间: {training_time:.2f}秒")
            
            # 可视化结果
            visualize_mlp_results(model, X_test, y_test, metrics, model_name)
            
            # 参数与拟合优度相关性分析
            plot_parameter_correlation(params_history, scores_history, model_name)
            
            # 优化收敛过程可视化
            plot_optimization_convergence(convergence_history, model_name)
            
            # 误差分析
            visualize_error_analysis(model, X_test, y_test, model_name)
    
    # 如果有多个优化方法，进行交叉验证
    if len(all_models) > 1:
        print("\n===== 对最佳模型进行交叉验证 =====")
        best_model_name = max(all_metrics, key=lambda k: all_metrics[k]['r2'])
        best_model = all_models[best_model_name]["model"]
        
        cv_results = perform_cross_validation(best_model, X_data, y_data, n_splits=5)
        print(f"交叉验证结果 ({best_model_name}):")
        print(f"  平均 R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
        print(f"  平均 RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
        print(f"  平均 MAE: {cv_results['mean_mae']:.4f} ± {cv_results['std_mae']:.4f}")
        
        # 绘制模型比较图
        plot_model_comparison(all_metrics, all_stability)
    else:
        # 直接对唯一的模型进行交叉验证
        print("\n===== 对多层感知机模型进行交叉验证 =====")
        model_name = list(all_models.keys())[0]
        model = all_models[model_name]["model"]
        
        cv_results = perform_cross_validation(model, X_data, y_data, n_splits=5)
        print(f"交叉验证结果 (多层感知机):")
        print(f"  平均 R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
        print(f"  平均 RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
        print(f"  平均 MAE: {cv_results['mean_mae']:.4f} ± {cv_results['std_mae']:.4f}")
    
    print("分析完成！")
