# For systems
import json

# For data processing
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

from ..utils import *
from .generic_model import GenericModelPipeline, GenericResultAggregator

class PolyPipeline(GenericModelPipeline):
    def __init__(self, model_name, score_name, poly_json, **kwargs):
        with open(poly_json, 'r') as file:
            json_content = json.load(file)
        
        self.param_space = None if "param_space" not in json_content else json_content["param_space"]
        self.pipeline_type = "basic" if "pipeline_type" not in json_content else json_content["pipeline_type"]
        self.degree = 1 if "degree" not in json_content else json_content["degree"] # Only for pipeline type basic

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
        X_train_copy = X_train[self.numerical_columns].copy(True)
        X_poly = X_train_copy.values
        for deg in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X_train_copy.values**deg))
        self.scaler.fit(X_poly)

    def scale_dataset(self, X_input):
        X_input_scaled = X_input[self.numerical_columns].copy(True)
        X_poly = X_input_scaled.values
        for deg in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X_input_scaled.values**deg))
        X_poly = self.scaler.transform(X_poly)
        if "eng_target" in X_input.columns:
            eng_col = np.reshape(X_input["eng_target"].values, (-1, 1))
            X_poly = np.hstack((X_poly, eng_col))
        return X_poly
    
    # Main pipeline for training the model
    def run_train(self, X_train, Y_train, score_se_df, list_X_tests, seed=RANDOM_SEED):
        self.setup_scaler(X_train)
        X_train_scaled = self.scale_dataset(X_train)
        
        if self.pipeline_type == "basic":
            # Initialize the model and perform cross-validation
            self.best_model = ElasticNet(alpha=0.1, l1_ratio=0.9)
            scores = cross_val_score(self.best_model, X_train_scaled, Y_train, n_jobs=N_JOBS,
                                    cv=KFold(n_splits=self.cv, shuffle=True, random_state=seed),
                                    scoring='neg_mean_squared_error')
            self.best_model.fit(X_train_scaled, Y_train)

            # Evaluate RMSE from cross-validation
            self.cv_rmse = np.sqrt(-np.mean(scores))  # Taking the negative to get positive RMSE
            self.aggregator.set_cv_rmse(self.cv_rmse)
        else:
            # Parameter for GridSearch
            param_grid = {
                'poly_features__degree': self.param_space["poly_degree"],   # Polynomial degree
                'elastic_net__alpha': self.param_space["alpha"],            # Regularization strength
                'elastic_net__l1_ratio': self.param_space["l1_ratio"]       # L1 ratio for ElasticNet
            }

            # Create a pipeline
            polynomial_pipeline = Pipeline([
                ('poly_features', PolynomialFeatures()),
                ('elastic_net', ElasticNet())
            ])

            grid_search = GridSearchCV(
                n_jobs=N_JOBS, estimator=polynomial_pipeline, param_grid=param_grid,
                cv=self.cv, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_scaled, Y_train)

            self.best_model = grid_search.best_estimator_

            # Get the best parameters
            self.best_params = grid_search.best_params_

            # Evaluate RMSE from cross-validation
            self.cv_rmse = np.sqrt(-grid_search.best_score_)  # Taking the negative to get positive RMSE
            
            self.aggregator.set_best_params(self.best_params)
            self.aggregator.set_cv_rmse(self.cv_rmse)

    def perform_prediction(self, X_test):
        # Predict missing values
        X_test_scaled = self.scale_dataset(X_test)
        Y_pred = self.best_model.predict(X_test_scaled)
        return Y_pred
