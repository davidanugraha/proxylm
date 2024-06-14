# For systems
import json
import logging

# For data processing
import numpy as np

# For models
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor

from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer

from ..utils import *
from .generic_model import GenericModelPipeline, GenericResultAggregator

class LGBMPipeline(GenericModelPipeline):
    def __init__(self, model_name, score_name, lgbm_json, **kwargs):
        with open(lgbm_json, 'r') as file:
            json_content = json.load(file)
        
        self.hpm_search_mode = json_content["hpm_search_mode"]
        self.fixed_param = json_content.get("fixed_param", {})
        
        if self.hpm_search_mode == "bayes_search":
            # Initialize BayesSearchCV through parameter space
            self.param_space = {}
            for param_name, param_info in json_content["param_space"].items():
                param_type = param_info["type"]
                low = param_info["low"]
                high = param_info["high"]
                prior = param_info["prior"] if "prior" in param_info else "uniform"
                if param_type == "Integer":
                    self.param_space[param_name] = Integer(low, high, prior=prior)
                elif param_type == "Real":
                    self.param_space[param_name] = Real(low, high, prior=prior)
                elif param_type == "Categorical":
                    step = param_info["step"] if "step" in param_info else 1
                    self.param_space[param_name] = Categorical(list(range(low, high + 1, step)))
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")
            self.n_iter = json_content["n_iter"]
        elif self.hpm_search_mode == "grid_search":
            # Initialize GridSearchCV through parameter space
            self.param_space = json_content["param_space"]
        else:
            raise ValueError(f"Unsupported hyperparameter search type: {self.hpm_search_mode}")
        
        # Cross validation
        self.cv = json_content["cv"]
        
        # Intitialize aggregator
        self.aggregator = GenericResultAggregator()
        
    def aggregate_result(self, test_result, X_columns):
        self.aggregator.aggregate_result(test_result, X_columns)
        
    def aggregate_result_lolo(self, test_source_result, test_target_result, X_columns):
        self.aggregator.aggregate_result_lolo(test_source_result, test_target_result, X_columns)
        
    def aggregate_result_list(self, test_result, test_column_names, X_columns):
        self.aggregator.aggregate_result_list(test_result, test_column_names, X_columns)
        
    def retrieve_aggregated_result(self, X_columns):
        return self.aggregator.retrieve_aggregated_result(X_columns)
    
    def retrieve_aggregated_result_lolo(self, X_columns):
        return self.aggregator.retrieve_aggregated_result_lolo(X_columns)
    
    def retrieve_aggregated_result_list(self, test_column_names, X_columns):
        return self.aggregator.retrieve_aggregated_result_list(test_column_names, X_columns)
    
    def setup_scaler(self, X_train):
        # Initialize the StandardScaler for predictors and for the target variable
        self.scaler = StandardScaler()
        self.numerical_columns = list(X_train.columns)

        # Identify the numerical indices
        if 'eng_target' in self.numerical_columns:
            self.numerical_columns.remove("eng_target")

        # Fit the numerical training data
        self.scaler.fit(X_train[self.numerical_columns])

    def scale_dataset(self, X_input):
        X_input_scaled = X_input.copy()
        X_input_scaled[self.numerical_columns] = self.scaler.transform(X_input[self.numerical_columns])
        return X_input_scaled
    
    # Main pipeline for training the model
    def run_train(self, X_train, Y_train, score_se_df, list_X_tests, seed=RANDOM_SEED):
        # Setup scaler for latter transformation
        self.setup_scaler(X_train)

        X_train_new = X_train.copy(True)
        Y_train_new = Y_train.copy(True)
        X_train_new = self.scale_dataset(X_train_new)
        
        # Initialize the LGBM model
        lgbm_model = LGBMRegressor(random_state=seed, **self.fixed_param)
        
        
        if self.hpm_search_mode == "bayes_search":
            logging.debug('Running LGBM Bayes Search')
            logging.debug(f"Parameter space: {self.param_space}")
            self.hpm_search_algorithm = BayesSearchCV(estimator=lgbm_model,
                                                    search_spaces=self.param_space,
                                                    scoring='neg_mean_squared_error',
                                                    cv=self.cv,
                                                    n_iter=self.n_iter,
                                                    random_state=seed,
                                                    n_jobs=N_JOBS)
        elif self.hpm_search_mode == "grid_search":
            logging.debug('Running LGBM Grid Search')
            logging.debug(f"Parameter grid: {self.param_space}")
            self.hpm_search_algorithm = GridSearchCV(estimator=lgbm_model,
                                                    param_grid=self.param_space,
                                                    scoring='neg_mean_squared_error',
                                                    cv=self.cv,
                                                    n_jobs=N_JOBS)

        # Fit the grid search to the data
        self.hpm_search_algorithm.fit(X_train_new, Y_train_new)

        # Get the best model
        self.best_model = self.hpm_search_algorithm.best_estimator_

        # Get the best parameters
        self.aggregator.set_best_params(self.hpm_search_algorithm.best_params_)

        # Evaluate RMSE from cross-validation, taking the negative to get positive RMSE
        self.aggregator.set_cv_rmse(np.sqrt(-self.hpm_search_algorithm.best_score_))  
        
        # Set feature importance
        self.aggregator.set_feature_importances(self.best_model.feature_importances_)

    def perform_prediction(self, X_test):
        # Predict missing values
        X_test_scaled = self.scale_dataset(X_test)
        Y_pred = self.best_model.predict(X_test_scaled)
        return Y_pred

