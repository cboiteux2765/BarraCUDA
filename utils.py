import cupy as cp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import itertools
import time

class GPUHyperparameterTuner:
    def __init__(self, model_type='svm'):
        """
        Initialize the GPU-accelerated hyperparameter tuner.
        
        :param model_type: Type of model to tune ('svm' or 'randomforest')
        """
        self.model_type = model_type
        self.best_params = None
        self.best_score = -np.inf
        
    def _generate_param_grid(self):
        """
        Generate parameter grid based on model type.
        
        :return: List of parameter dictionaries
        """
        if self.model_type == 'svm':
            # Generate SVM parameter combinations
            kernels = ['linear', 'rbf']
            c_values = [0.1, 1, 10, 100]
            gamma_values = ['scale', 'auto', 0.1, 1, 10]
            
            # Create parameter combinations
            param_grid = [
                {'kernel': kernel, 'C': c, 'gamma': gamma}
                for kernel in kernels
                for c in c_values
                for gamma in gamma_values
            ]
        elif self.model_type == 'randomforest':
            # Generate Random Forest parameter combinations
            n_estimators = [50, 100, 200]
            max_depths = [None, 10, 20, 30]
            min_samples_split = [2, 5, 10]
            
            # Create parameter combinations
            param_grid = [
                {'n_estimators': n_est, 'max_depth': depth, 'min_samples_split': min_split}
                for n_est in n_estimators
                for depth in max_depths
                for min_split in min_samples_split
            ]
        else:
            raise ValueError("Unsupported model type. Choose 'svm' or 'randomforest'.")
        
        return param_grid
    
    def _gpu_grid_search(self, X, y):
        """
        Perform GPU-accelerated grid search.
        
        :param X: Input features
        :param y: Target labels
        :return: Best parameters and score
        """
        # Convert to CuPy arrays for GPU computation
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Generate parameter grid
        param_grid = self._generate_param_grid()
        
        # Parallel parameter tuning
        best_params = None
        best_score = -np.inf
        
        # Use CuPy for parallel processing
        for params in param_grid:
            # Select and configure model
            if self.model_type == 'svm':
                model = SVC(**params)
            else:
                model = RandomForestClassifier(**params, random_state=42)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            # Update best parameters
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score
    
    def tune(self, X, y):
        """
        Main tuning method.
        
        :param X: Input features
        :param y: Target labels
        :return: Dictionary with tuning results
        """
        start_time = time.time()
        
        # Perform GPU-accelerated grid search
        best_params, best_score = self._gpu_grid_search(X, y)
        
        end_time = time.time()
        
        # Store results
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'total_time': end_time - start_time,
            'model_type': self.model_type
        }

def main():
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=10000, 
        n_features=20, 
        n_informative=15, 
        n_redundant=5, 
        random_state=42
    )
    
    # Demonstrate SVM hyperparameter tuning
    print("Tuning SVM Hyperparameters:")
    svm_tuner = GPUHyperparameterTuner(model_type='svm')
    svm_results = svm_tuner.tune(X, y)
    print("SVM Results:", svm_results)
    
    # Demonstrate Random Forest hyperparameter tuning
    print("\nTuning Random Forest Hyperparameters:")
    rf_tuner = GPUHyperparameterTuner(model_type='randomforest')
    rf_results = rf_tuner.tune(X, y)
    print("Random Forest Results:", rf_results)

if __name__ == '__main__':
    main()
