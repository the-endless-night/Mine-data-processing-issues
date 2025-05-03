import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import KFold, cross_val_score
import warnings

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_pred = self.model.predict(X_test)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # 计算基本指标
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # 计算均方对数误差 (MSLE)
        # 避免负值和零值
        y_test_pos = np.maximum(y_test, 1e-10)
        y_pred_pos = np.maximum(y_pred, 1e-10)
        
        try:
            msle = mean_squared_log_error(y_test_pos, y_pred_pos)
        except:
            warnings.warn("无法计算均方对数误差，可能存在负值")
            msle = np.nan
        
        # 计算平均相对误差 (MRE)
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_errors = np.abs(y_test - y_pred) / np.abs(y_test)
            mre = 100 * np.nanmean(relative_errors)  # 百分比形式
        
        # 计算决定系数调整值 (Adjusted R²)
        n = len(y_test)
        p = X_test.shape[1] if X_test.ndim > 1 else 1
        if n - p - 1 > 0:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = np.nan
        
        metrics = {
            "r2": r2,
            "adjusted_r2": adj_r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "msle": msle,
            "mre": mre
        }
        
        return metrics
    
    def evaluate_stability(self, X, y, n_splits=5):
        """评估模型稳定性"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        r2_scores = []
        rmse_scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 重新训练模型
            if hasattr(self.model, 'fit'):
                if hasattr(self.model, 'degree'):  # 处理多项式回归特例
                    degree = self.model.degree
                    poly_features = self.model.poly
                    model = type(self.model)(degree=degree)
                    model.fit(X_train, y_train)
                else:
                    # 深拷贝模型并重新训练
                    from sklearn.base import clone
                    model = clone(self.model)
                    if y_train.ndim > 1:
                        model.fit(X_train, y_train.ravel())
                    else:
                        model.fit(X_train, y_train)
                
                # 预测并计算R²和RMSE
                y_pred = model.predict(X_test)
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2_scores.append(r2)
                rmse_scores.append(rmse)
        
        # 计算稳定性指标
        # 1. R²的标准差越小，模型越稳定
        # 2. RMSE的标准差越小，模型越稳定
        # 3. 结合两个指标
        r2_stability = 1.0 / (1.0 + np.std(r2_scores))
        rmse_stability = 1.0 / (1.0 + np.std(rmse_scores))
        
        # 综合稳定性指标
        stability = 0.6 * r2_stability + 0.4 * rmse_stability
        
        return stability
    
    def evaluate_applicability(self, X, domain_min, domain_max):
        """评估模型在特定域范围内的适用性"""
        # 检查X是否在定义域内
        in_domain = np.all((X >= domain_min) & (X <= domain_max), axis=1)
        applicability = np.mean(in_domain)
        return applicability

def perform_cross_validation(model, X, y, n_splits=5):
    """对模型进行交叉验证"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 克隆并训练模型
        from sklearn.base import clone
        cv_model = clone(model)
        
        if hasattr(cv_model, 'fit'):
            if y_train.ndim > 1:
                cv_model.fit(X_train, y_train.ravel())
            else:
                cv_model.fit(X_train, y_train)
            
            # 预测并计算指标
            y_pred = cv_model.predict(X_test)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
    
    # 计算统计值
    cv_results = {
        "r2_scores": r2_scores,
        "mean_r2": np.mean(r2_scores),
        "std_r2": np.std(r2_scores),
        "rmse_scores": rmse_scores,
        "mean_rmse": np.mean(rmse_scores),
        "std_rmse": np.std(rmse_scores),
        "mae_scores": mae_scores,
        "mean_mae": np.mean(mae_scores),
        "std_mae": np.std(mae_scores)
    }
    
    return cv_results
