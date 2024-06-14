import logging

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from ..utils import *

class GenericModelPipeline(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run_train(self, X_train, Y_train, score_se_df, list_X_tests, seed=RANDOM_SEED):
        """
        Train the model using the preprocessed data with whatever training algorithm used.

        Parameters:
        - X_train: Input features
        - Y_train: Target labels
        - score_se_df: For augmentation
        - X_test: For augmentationA
        """
        pass

    @abstractmethod
    def perform_prediction(self, X_test):
        """
        Make predictions using the trained model.

        Parameters:
        - X_test: Test input features

        Returns:
        - Predicted values (y_test)
        """
        pass
    
    # Lowkey, the following are bad interfaces, need to refactor some day
    @abstractmethod
    def aggregate_result(self, test_result, X_columns):
        pass
    
    @abstractmethod
    def aggregate_result_lolo(self, test_source_result, test_target_result, X_columns):
        pass
        
    @abstractmethod
    def aggregate_result_list(self, test_result, test_column_names, X_columns):
        pass
        
    @abstractmethod    
    def retrieve_aggregated_result(self, X_columns):
        pass
        
    @abstractmethod
    def retrieve_aggregated_result_lolo(self, X_columns):
        pass
        
    @abstractmethod
    def retrieve_aggregated_result_list(self, test_column_names, X_columns):
        pass

class GenericResultAggregator():
    def __init__(self):
        self.aggr_result = {}
        self.best_params = None
        self.feature_importances = None
        self.cv_rmse = None

    def get_aggr_result(self):
        return self.aggr_result
    
    def get_best_params(self):
        return self.best_params
    
    def get_feature_importances(self):
        return self.feature_importances
    
    def get_cv_rmse(self):
        return self.cv_rmse
    
    def set_best_params(self, best_params):
        self.best_params = best_params
        
    def set_feature_importances(self, feature_importances):
        self.feature_importances = feature_importances
        
    def set_cv_rmse(self, cv_rmse):
        self.cv_rmse = cv_rmse
    
    def aggregate_result(self, test_result, X_columns):
        # Get feature importances
        if len(self.aggr_result) == 0:
            # Have to initialize the columns
            self.aggr_result["cv_rmse"] = [self.cv_rmse]
            self.aggr_result["test_rmse"] = [test_result]
            
            # Add feature importances
            if self.feature_importances is not None:
                for feature, importance in zip(X_columns, self.feature_importances):
                    self.aggr_result[f'{feature}_importance'] = [importance]

            # Add best_params to the dictionary as separate columns
            if self.best_params:
                for param, value in self.best_params.items():
                    self.aggr_result[param] = [value]
        else:
            # Simply append to the existing dictionary
            self.aggr_result["cv_rmse"].append(self.cv_rmse)
            self.aggr_result["test_rmse"].append(test_result)
            
            # Add feature importances
            if self.feature_importances is not None:
                for feature, importance in zip(X_columns, self.feature_importances):
                    self.aggr_result[f'{feature}_importance'].append(importance)

            # Add best_params to the dictionary as separate columns
            if self.best_params:
                for param, value in self.best_params.items():
                    self.aggr_result[param].append(value)
        
        # Print the best parameters and RMSE
        logging.debug(f'- Best Parameters: {self.best_params}')
        logging.debug(f'- Cross-Validated RMSE: {self.cv_rmse}')
        logging.debug(f'- Test RMSE: {test_result}')
        
    def aggregate_result_lolo(self, test_source_result, test_target_result, X_columns):
        if len(self.aggr_result) == 0:
            # Have to initialize the columns
            self.aggr_result["cv_rmse"] = [self.cv_rmse]
            self.aggr_result["test_source_rmse"] = [test_source_result]
            self.aggr_result["test_target_rmse"] = [test_target_result]
            
            # Add feature importances
            if self.feature_importances is not None:
                for feature, importance in zip(X_columns, self.feature_importances):
                    self.aggr_result[f'{feature}_importance'] = [importance]

            # Add best_params to the dictionary as separate columns
            if self.best_params:
                for param, value in self.best_params.items():
                    self.aggr_result[param] = [value]
        else:
            # Simply append to the existing dictionary
            self.aggr_result["cv_rmse"].append(self.cv_rmse)
            self.aggr_result["test_source_rmse"].append(test_source_result)
            self.aggr_result["test_target_rmse"].append(test_target_result)
            
            # Add feature importances
            if self.feature_importances is not None:
                for feature, importance in zip(X_columns, self.feature_importances):
                    self.aggr_result[f'{feature}_importance'].append(importance)

            # Add best_params to the dictionary as separate columns
            if self.best_params:
                for param, value in self.best_params.items():
                    self.aggr_result[param].append(value)
        
        # Print the best parameters and RMSE
        logging.debug(f'- Best Parameters: {self.best_params}')
        logging.debug(f'- Cross-Validated RMSE: {self.cv_rmse}')
        logging.debug(f'- Test source RMSE: {test_source_result}')
        logging.debug(f'- Test target RMSE: {test_target_result}')
        
    def aggregate_result_list(self, test_result, test_column_names, X_columns):
        if len(self.aggr_result) == 0:
            # Have to initialize the columns
            self.aggr_result["cv_rmse"] = [self.cv_rmse]
            for i, test_name in enumerate(test_column_names):
                self.aggr_result[test_name] = [test_result[i]]
            
            # Add feature importances
            if self.feature_importances is not None:
                for feature, importance in zip(X_columns, self.feature_importances):
                    self.aggr_result[f'{feature}_importance'] = [importance]

            # Add best_params to the dictionary as separate columns
            if self.best_params:
                for param, value in self.best_params.items():
                    self.aggr_result[param] = [value]
        else:
            # Simply append to the existing dictionary
            self.aggr_result["cv_rmse"].append(self.cv_rmse)
            for i, test_name in enumerate(test_column_names):
                self.aggr_result[test_name].append(test_result[i])
            
            # Add feature importances
            if self.feature_importances is not None:
                for feature, importance in zip(X_columns, self.feature_importances):
                    self.aggr_result[f'{feature}_importance'].append(importance)

            # Add best_params to the dictionary as separate columns
            if self.best_params:
                for param, value in self.best_params.items():
                    self.aggr_result[param].append(value)
        
        # Print the best parameters and RMSE
        logging.debug(f'- Best Parameters: {self.best_params}')
        logging.debug(f'- Cross-Validated RMSE: {self.cv_rmse}')
        for i, test_name in enumerate(test_column_names):
            logging.debug(f'- Test RMSE for {test_name}: {test_result[i]}')
        
    def retrieve_aggregated_result(self, X_columns):
        # Average the RMSE 
        self.aggr_result["cv_rmse_se"] = [np.std(self.aggr_result["cv_rmse"])]
        self.aggr_result["test_rmse_se"] = [np.std(self.aggr_result["test_rmse"])]
        self.aggr_result["cv_rmse"] = [np.mean(self.aggr_result["cv_rmse"])]
        self.aggr_result["test_rmse"] = [np.mean(self.aggr_result["test_rmse"])]
        
        # Average feature importances
        if self.feature_importances is not None:
            for feature, _ in zip(X_columns, self.feature_importances):
                self.aggr_result[f'{feature}_importance'] = [np.mean(self.aggr_result[f'{feature}_importance'])]

        # Average best_params
        if self.best_params:
            for param, _ in self.best_params.items():
                self.aggr_result[param] = [np.mean(self.aggr_result[param])]
        
        return pd.DataFrame(self.aggr_result)
    
    def retrieve_aggregated_result_lolo(self, X_columns):
        # Average the RMSE
        self.aggr_result["cv_rmse_se"] = [np.std(self.aggr_result["cv_rmse"])]
        self.aggr_result["test_source_rmse_se"] = [np.std(self.aggr_result["test_source_rmse"])]
        self.aggr_result["test_target_rmse_se"] = [np.std(self.aggr_result["test_target_rmse"])]
        self.aggr_result["cv_rmse"] = [np.mean(self.aggr_result["cv_rmse"])]
        self.aggr_result["test_source_rmse"] = [np.mean(self.aggr_result["test_source_rmse"])]
        self.aggr_result["test_target_rmse"] = [np.mean(self.aggr_result["test_target_rmse"])]
        
        # Average feature importances
        if self.feature_importances is not None:
            for feature, _ in zip(X_columns, self.feature_importances):
                self.aggr_result[f'{feature}_importance'] = [np.mean(self.aggr_result[f'{feature}_importance'])]

        # Average best_params
        if self.best_params:
            for param, _ in self.best_params.items():
                self.aggr_result[param] = [np.mean(self.aggr_result[param])]
        
        return pd.DataFrame(self.aggr_result)
    
    def retrieve_aggregated_result_list(self, test_column_names, X_columns):
        # Average the RMSE
        self.aggr_result["test_source_rmse"] = []
        self.aggr_result["test_target_rmse"] = []
        self.aggr_result["cv_rmse_se"] = [np.std(self.aggr_result["cv_rmse"])]
        self.aggr_result["cv_rmse"] = [np.mean(self.aggr_result["cv_rmse"])]
        for i, test_name in enumerate(test_column_names):
            self.aggr_result[f"{test_name}_se"] = [np.std(self.aggr_result[test_name])]
            self.aggr_result[test_name] = [np.mean(self.aggr_result[test_name])]
            if test_name.startswith("test_source"):
                self.aggr_result["test_source_rmse"].append(self.aggr_result[test_name])
            else:
                self.aggr_result["test_target_rmse"].append(self.aggr_result[test_name])
        self.aggr_result["test_source_rmse"] = [np.mean(self.aggr_result["test_source_rmse"])]
        self.aggr_result["test_target_rmse"] = [np.mean(self.aggr_result["test_target_rmse"])]
        
        # Average feature importances
        if self.feature_importances is not None:
            for feature, _ in zip(X_columns, self.feature_importances):
                self.aggr_result[f'{feature}_importance'] = [np.mean(self.aggr_result[f'{feature}_importance'])]

        # Average best_params
        if self.best_params:
            for param, _ in self.best_params.items():
                self.aggr_result[param] = [np.mean(self.aggr_result[param])]
        
        return pd.DataFrame(self.aggr_result)
