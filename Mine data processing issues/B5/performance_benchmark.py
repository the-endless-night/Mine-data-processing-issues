import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import psutil
import argparse

def get_memory_info():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # 转换为MB

def get_gpu_memory_info():
    """获取当前GPU内存使用情况，如果可用"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)  # 转换为MB
    return 0

def benchmark_function(func, *args, **kwargs):
    """测量函数执行时间和内存使用情况"""
    # 记录开始时间
    start_time = time.time()
    start_memory = get_memory_info()
    start_gpu_memory = get_gpu_memory_info()
    
    # 执行函数
    result = func(*args, **kwargs)
    
    # 记录结束时间和内存使用
    end_time = time.time()
    end_memory = get_memory_info()
    end_gpu_memory = get_gpu_memory_info()
    
    # 计算差异
    elapsed_time = end_time - start_time
    memory_used = end_memory - start_memory
    gpu_memory_used = end_gpu_memory - start_gpu_memory
    
    return {
        'result': result,
        'time': elapsed_time,
        'cpu_memory': memory_used,
        'gpu_memory': gpu_memory_used
    }

def compare_performance():
    """比较优化前后的性能差异"""
    from main import train_autoencoder, reduce_and_reconstruct
    import numpy as np
    
    # 创建测试数据
    input_dim = 100
    n_samples = 1000
    encoding_dim = 20
    X_data = np.random.randn(n_samples, input_dim)
    
    # 设置测试批量大小
    batch_sizes = [32, 64, 128, 256]
    
    results = {
        'batch_size': [],
        'training_time': [],
        'encoding_time': [],
        'reconstruction_time': [],
        'memory_usage': [],
        'gpu_memory_usage': []
    }
    
    for batch_size in batch_sizes:
        print(f"测试批量大小: {batch_size}")
        
        # 训练自编码器
        benchmark = benchmark_function(
            train_autoencoder, 
            X_data, 
            encoding_dim=encoding_dim, 
            epochs=10,  # 快速测试用的小epoch数
            batch_size=batch_size
        )
        
        autoencoder = benchmark['result'][0]
        training_time = benchmark['time']
        memory_usage = benchmark['cpu_memory']
        gpu_memory_usage = benchmark['gpu_memory']
        
        # 测试编码和重构时间
        encoding_benchmark = benchmark_function(reduce_and_reconstruct, X_data, autoencoder)
        _, _, encoding_time, reconstruction_time = encoding_benchmark['result']
        
        # 收集结果
        results['batch_size'].append(batch_size)
        results['training_time'].append(training_time)
        results['encoding_time'].append(encoding_time)
        results['reconstruction_time'].append(reconstruction_time)
        results['memory_usage'].append(memory_usage)
        results['gpu_memory_usage'].append(gpu_memory_usage)
        
        print(f"  训练时间: {training_time:.2f}秒")
        print(f"  编码时间: {encoding_time:.4f}秒")
        print(f"  重构时间: {reconstruction_time:.4f}秒")
        print(f"  CPU内存使用: {memory_usage:.2f}MB")
        if torch.cuda.is_available():
            print(f"  GPU内存使用: {gpu_memory_usage:.2f}MB")
        print()
    
    # 可视化结果
    plt.figure(figsize=(16, 10))
    
    # 训练时间
    plt.subplot(2, 2, 1)
    plt.plot(results['batch_size'], results['training_time'], 'o-')
    plt.title('批量大小 vs 训练时间')
    plt.xlabel('批量大小')
    plt.ylabel('训练时间 (秒)')
    plt.grid(True, alpha=0.3)
    
    # 编码和重构时间
    plt.subplot(2, 2, 2)
    plt.plot(results['batch_size'], results['encoding_time'], 'o-', label='编码时间')
    plt.plot(results['batch_size'], results['reconstruction_time'], 's-', label='重构时间')
    plt.title('批量大小 vs 推理时间')
    plt.xlabel('批量大小')
    plt.ylabel('时间 (秒)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 内存使用
    plt.subplot(2, 2, 3)
    plt.plot(results['batch_size'], results['memory_usage'], 'o-')
    plt.title('批量大小 vs CPU内存使用')
    plt.xlabel('批量大小')
    plt.ylabel('内存使用 (MB)')
    plt.grid(True, alpha=0.3)
    
    # GPU内存使用
    if torch.cuda.is_available():
        plt.subplot(2, 2, 4)
        plt.plot(results['batch_size'], results['gpu_memory_usage'], 'o-')
        plt.title('批量大小 vs GPU内存使用')
        plt.xlabel('批量大小')
        plt.ylabel('GPU内存 (MB)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存结果
    output_dir = 'd:/amath/B/B5/output'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{output_dir}/performance_benchmark_{timestamp}.png')
    plt.close()
    
    # 计算并返回最佳批量大小
    throughput = [bs / rt for bs, rt in zip(results['batch_size'], results['training_time'])]
    best_idx = np.argmax(throughput)
    best_batch_size = results['batch_size'][best_idx]
    
    print(f"性能分析完成。最佳批量大小: {best_batch_size} (吞吐量: {throughput[best_idx]:.2f}样本/秒)")
    return best_batch_size

def measure_amdahl_speedup():
    """基于Amdahl定律评估加速比"""
    # 创建测试数据和简单模型
    input_dim, n_samples = 100, 1000
    X_data = np.random.randn(n_samples, input_dim).astype(np.float32)
    
    # 测试没有优化的执行时间
    def baseline_run():
        X_tensor = torch.FloatTensor(X_data)
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim)
        )
        for _ in range(10):
            out = model(X_tensor)
            loss = torch.mean((out - X_tensor) ** 2)
            loss.backward()
        return
    
    # 测试优化后的执行时间（混合精度）
    def optimized_run():
        X_tensor = torch.FloatTensor(X_data).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim)
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        if torch.cuda.is_available():
            # 从torch.amp导入而不是torch.cuda.amp
            from torch.amp import GradScaler, autocast
            # 使用新的API格式
            scaler = GradScaler('cuda')
            
            for _ in range(10):
                # 使用正确的autocast调用方式
                with autocast('cuda'):
                    out = model(X_tensor)
                    loss = torch.mean((out - X_tensor) ** 2)
                scaler.scale(loss).backward()
                scaler.step(lambda: None)  # 空操作
                scaler.update()
        else:
            for _ in range(10):
                out = model(X_tensor)
                loss = torch.mean((out - X_tensor) ** 2)
                loss.backward()
        return
    
    # 测量时间
    baseline_time = benchmark_function(baseline_run)['time']
    optimized_time = benchmark_function(optimized_run)['time']
    
    # 计算加速比
    speedup = baseline_time / optimized_time if optimized_time > 0 else float('inf')
    
    print(f"Amdahl定律加速分析:")
    print(f"基准时间: {baseline_time:.4f}秒")
    print(f"优化时间: {optimized_time:.4f}秒")
    print(f"实际加速比: {speedup:.2f}倍")
    
    # 使用Amdahl公式计算理论加速比
    # 假设可并行部分p=0.8，加速比k=2
    p = 0.8  # 可并行部分比例
    k = 2.0  # 该部分加速倍数
    theoretical_speedup = 1 / ((1 - p) + p/k)
    
    print(f"假设80%代码可优化且加速2倍:")
    print(f"理论加速比: {theoretical_speedup:.2f}倍")
    print(f"实际和理论的比值: {speedup/theoretical_speedup:.2f}")
    
    return {
        'baseline_time': baseline_time,
        'optimized_time': optimized_time,
        'actual_speedup': speedup,
        'theoretical_speedup': theoretical_speedup
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='性能基准测试')
    parser.add_argument('--test', type=str, default='all', choices=['batch', 'amdahl', 'all'],
                      help='要运行的测试类型: batch (批量大小优化), amdahl (Amdahl定律分析), all (全部)')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs('d:/amath/B/B5/output', exist_ok=True)
    
    if args.test in ['batch', 'all']:
        print("== 批量大小性能测试 ==")
        best_batch_size = compare_performance()
    
    if args.test in ['amdahl', 'all']:
        print("\n== Amdahl定律加速比分析 ==")
        amdahl_results = measure_amdahl_speedup()
