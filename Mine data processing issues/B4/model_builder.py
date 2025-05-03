import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time

warnings.filterwarnings('ignore')

class AdaptiveOptimizer:
    """参数自适应优化器"""
    
    def __init__(self, param_grid, evaluation_metric=r2_score, method='bayesian'):
        self.param_grid = param_grid
        self.evaluation_metric = evaluation_metric
        self.best_params = None
        self.best_score = -np.inf
        self.params_history = []
        self.scores_history = []
        self.convergence_history = []
        self.method = method
        
    def optimize(self, model_class, X_train, y_train, X_val, y_val, n_iterations=50, early_stopping=5):
        """根据选择的方法优化参数"""
        print(f"使用{self.method}优化方法进行参数搜索...")
        
        if self.method == 'bayesian':
            return self.bayesian_optimize(model_class, X_train, y_train, X_val, y_val, n_iterations, early_stopping)
        elif self.method == 'genetic':
            return self.genetic_optimize(model_class, X_train, y_train, X_val, y_val, n_iterations, early_stopping)
        elif self.method == 'random':
            return self.random_search(model_class, X_train, y_train, X_val, y_val, n_iterations, early_stopping)
        else:
            print(f"未知的优化方法: {self.method}，使用贝叶斯优化作为默认方法")
            return self.bayesian_optimize(model_class, X_train, y_train, X_val, y_val, n_iterations, early_stopping)
    
    def bayesian_optimize(self, model_class, X_train, y_train, X_val, y_val, n_iterations=50, early_stopping=5):
        """增强的贝叶斯优化算法寻找最优参数"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
        import random
        
        # 初始随机采样点
        param_keys = list(self.param_grid.keys())
        n_init_points = min(10, n_iterations // 2)
        
        # 初始化高斯过程回归器 - 使用更复杂的核函数
        kernel = Matern(nu=2.5) + RBF() + WhiteKernel(noise_level=1e-5)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10)
        
        # 记录最佳分数的停滞次数，用于早停
        stagnation_count = 0
        prev_best = -np.inf
        
        # 初始随机采样和评估
        for _ in range(n_init_points):
            params = {k: random.choice(self.param_grid[k]) for k in param_keys}
            start_time = time.time()
            score = self._evaluate_params(model_class, params, X_train, y_train, X_val, y_val)
            eval_time = time.time() - start_time
            
            self.params_history.append(params)
            self.scores_history.append(score)
            self.convergence_history.append((_, score))
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            if stagnation_count >= early_stopping and early_stopping > 0:
                print(f"早停: {early_stopping}次迭代内无改进")
                break
        
        # 贝叶斯优化迭代
        for i in range(n_init_points, n_iterations):
            # 将参数编码为数值向量以用于GPR
            X_observed = []
            for params in self.params_history:
                x = [self.param_grid[k].index(params[k]) / (len(self.param_grid[k]) - 1) if len(self.param_grid[k]) > 1 else 0.5 for k in param_keys]
                X_observed.append(x)
            
            X_observed = np.array(X_observed)
            y_observed = np.array(self.scores_history)
            
            # 训练高斯过程
            gpr.fit(X_observed, y_observed)
            
            # 探索-利用平衡策略
            # 随机采样候选点
            n_samples = 1000
            X_samples = np.random.random((n_samples, len(param_keys)))
            
            # 计算采集函数值 (Expected Improvement)
            y_mean, y_std = gpr.predict(X_samples, return_std=True)
            # β参数控制探索-利用平衡，逐渐减少探索
            beta = max(0.1, 1.0 - i / n_iterations)
            acquisition = y_mean + beta * y_std
            
            best_idx = np.argmax(acquisition)
            
            # 解码参数
            next_params = {}
            for j, k in enumerate(param_keys):
                if len(self.param_grid[k]) > 1:
                    idx = int(round(X_samples[best_idx, j] * (len(self.param_grid[k]) - 1)))
                    next_params[k] = self.param_grid[k][idx]
                else:
                    next_params[k] = self.param_grid[k][0]
            
            # 评估新参数
            start_time = time.time()
            score = self._evaluate_params(model_class, next_params, X_train, y_train, X_val, y_val)
            eval_time = time.time() - start_time
            
            self.params_history.append(next_params)
            self.scores_history.append(score)
            self.convergence_history.append((i, score))
            
            # 更新最佳参数和早停计数
            if score > self.best_score:
                self.best_score = score
                self.best_params = next_params
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            if stagnation_count >= early_stopping and early_stopping > 0:
                print(f"早停: {early_stopping}次迭代内无改进")
                break
        
        return self.best_params, self.params_history, self.scores_history, self.convergence_history
    
    def genetic_optimize(self, model_class, X_train, y_train, X_val, y_val, n_iterations=50, early_stopping=5):
        """遗传算法优化参数"""
        import random
        
        param_keys = list(self.param_grid.keys())
        population_size = 20
        elite_size = 5
        mutation_rate = 0.1
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = {k: random.choice(self.param_grid[k]) for k in param_keys}
            population.append(individual)
        
        stagnation_count = 0
        
        # 进化迭代
        for i in range(n_iterations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                if individual not in self.params_history:
                    score = self._evaluate_params(model_class, individual, X_train, y_train, X_val, y_val)
                    self.params_history.append(individual)
                    self.scores_history.append(score)
                    self.convergence_history.append((i, score))
                else:
                    idx = self.params_history.index(individual)
                    score = self.scores_history[idx]
                
                fitness_scores.append(score)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = individual
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            
            if stagnation_count >= early_stopping and early_stopping > 0:
                print(f"早停: {early_stopping}次迭代内无改进")
                break
            
            # 选择精英个体
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
            elite = sorted_population[:elite_size]
            
            # 交叉和变异生成新种群
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                # 选择父代
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                
                # 交叉
                child = {}
                for k in param_keys:
                    child[k] = parent1[k] if random.random() < 0.5 else parent2[k]
                
                # 变异
                for k in param_keys:
                    if random.random() < mutation_rate:
                        child[k] = random.choice(self.param_grid[k])
                
                new_population.append(child)
            
            population = new_population
        
        return self.best_params, self.params_history, self.scores_history, self.convergence_history
    
    def random_search(self, model_class, X_train, y_train, X_val, y_val, n_iterations=50, early_stopping=5):
        """随机搜索优化参数"""
        import random
        
        param_keys = list(self.param_grid.keys())
        stagnation_count = 0
        
        for i in range(n_iterations):
            # 随机生成参数
            params = {k: random.choice(self.param_grid[k]) for k in param_keys}
            
            # 评估
            score = self._evaluate_params(model_class, params, X_train, y_train, X_val, y_val)
            self.params_history.append(params)
            self.scores_history.append(score)
            self.convergence_history.append((i, score))
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            if stagnation_count >= early_stopping and early_stopping > 0:
                print(f"早停: {early_stopping}次迭代内无改进")
                break
        
        return self.best_params, self.params_history, self.scores_history, self.convergence_history
    
    def _evaluate_params(self, model_class, params, X_train, y_train, X_val, y_val):
        """评估特定参数下模型的性能"""
        try:
            model = model_class(**params)
            model.fit(X_train, y_train.ravel())
            y_pred = model.predict(X_val).reshape(-1, 1)
            score = self.evaluation_metric(y_val, y_pred)
            return score
        except Exception as e:
            print(f"参数评估失败: {e}")
            return -np.inf


class ModelBuilder:
    """多层感知机模型构建器，专注于神经网络模型的自适应参数优化"""
    
    def __init__(self, model_type='mlp', optimization_method='bayesian'):
        # 选择模型类型
        self.model_type = model_type
        self.optimization_method = optimization_method
        
        if model_type == 'mlp':
            self.model_class = MLPRegressor
            self.param_grid = {
                "hidden_layer_sizes": [(50,), (100,), (150,), (200,), (50, 25), (100, 50), (150, 75), (100, 50, 25), 
                                      (200, 100), (200, 100, 50), (300, 150, 75)],
                "activation": ['relu', 'tanh', 'logistic'],
                "solver": ['adam', 'sgd', 'lbfgs'],
                "alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
                "learning_rate": ['constant', 'adaptive', 'invscaling'],
                "max_iter": [500, 1000, 2000, 3000]
            }
        elif model_type == 'rf':
            self.model_class = RandomForestRegressor
            self.param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ['auto', 'sqrt', 'log2']
            }
        elif model_type == 'gbdt':
            self.model_class = GradientBoostingRegressor
            self.param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.8, 0.9, 1.0]
            }
        else:
            print(f"未知的模型类型: {model_type}，使用多层感知机作为默认模型")
            self.model_type = 'mlp'
            self.model_class = MLPRegressor
            self.param_grid = {
                "hidden_layer_sizes": [(50,), (100,), (150,), (200,), (50, 25), (100, 50), (150, 75), (100, 50, 25)],
                "activation": ['relu', 'tanh', 'logistic'],
                "solver": ['adam', 'sgd', 'lbfgs'],
                "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
                "learning_rate": ['constant', 'adaptive', 'invscaling'],
                "max_iter": [500, 1000, 2000]
            }
    
    def build_and_optimize(self, X_train, y_train, X_test, y_test, n_iterations=50, early_stopping=5):
        """构建并优化模型"""
        print(f"正在训练{self.model_type}模型...")
        
        # 使用自适应参数优化
        optimizer = AdaptiveOptimizer(self.param_grid, method=self.optimization_method)
        best_params, params_history, scores_history, convergence_history = optimizer.optimize(
            self.model_class, X_train, y_train, X_test, y_test, 
            n_iterations=n_iterations, early_stopping=early_stopping
        )
        
        print(f"最佳参数: {best_params}")
        print(f"最佳R²分数: {optimizer.best_score:.4f}")
        
        # 使用最佳参数创建模型并训练
        model = self.model_class(**best_params)
        model.fit(X_train, y_train.ravel())
        
        return model, params_history, scores_history, convergence_history

# 多项式回归模型
class PolynomialRegressor:
    """多项式回归模型包装器"""
    
    def __init__(self, degree=2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()
    
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        return self
    
    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)
