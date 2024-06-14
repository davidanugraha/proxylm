##################################################
### Code adapted from Viktoria Schram (PP Bayesian)
#################################################

# For systems
import json
import logging
from copy import deepcopy
import itertools
import random

# For data processing
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

from ..utils import *
from .generic_model import GenericModelPipeline, GenericResultAggregator

class MFRegressor():
    def __init__(self, K, context_dict, context_info_len, random_state=RANDOM_SEED,
                 alpha=0.1, beta_w=0.1, beta_h=0.1, beta_z=0.01, beta_s=0.01, beta_t=0.01,
                 lr_decay=0.001, iterations=2000, verbose=False, **kwargs):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - K (int)                      : number of latent dimensions
        - alpha (float)                : learning rate
        - beta (float)                 : regularization parameter
        - context_dict (dict)          : context information
        - context_info_len(int)        : context information length
        """
        self.K = K
        self.alpha = alpha
        self.beta_h = beta_h
        self.beta_w = beta_w
        self.beta_z = beta_z
        self.beta_s = beta_s
        self.beta_t = beta_t
        self.iterations = iterations
        self.context_dict = context_dict
        self.context_info_len = context_info_len
        self.lr_decay = lr_decay
        self.random_state = random_state
        self.verbose = verbose
        
    def generate_score_matrix(self, X_train_data, langs_matrix, col_score_name):
        score_matrix = deepcopy(langs_matrix)

        for index, record in X_train_data.iterrows():
            src_lang = record["source_lang"]
            tgt_lang = record["target_lang"]
            dataset_size = str(int(record["dataset_size"]))
            score = record[col_score_name]
            score_matrix.loc[f"{src_lang}_{dataset_size}", tgt_lang] = score

        return score_matrix

    def fit(self, X_train_data, Y_train_data, X_dev_data, Y_dev_data, col_score_name, langs_matrix):
        # Create rating matrix based on X_train_data
        rating_matrix = self.generate_score_matrix(pd.concat([X_train_data, Y_train_data], axis=1),
                                                   langs_matrix, col_score_name)
        rating_matrix = rating_matrix.fillna(0)
        
        self.prediction = deepcopy(np.array(rating_matrix))
        self.src_langs = rating_matrix.index.tolist()
        self.tgt_langs = rating_matrix.columns.tolist()
        self.num_src, self.num_tgt = rating_matrix.shape
        self.num_scores_source = np.ones(self.num_src)
        self.num_scores_target = np.ones(self.num_tgt)
        rating_matrix = np.array(rating_matrix)
        
        # Initialize user and item latent feature matrice
        self.W = np.random.normal(scale=1. / self.K, size=(self.num_src, self.K))
        self.H = np.random.normal(scale=1. / self.K, size=(self.num_tgt, self.K))
        self.C = np.random.normal(scale=1. / self.context_info_len, size=self.context_info_len)
                
        # Initialize the biases
        # the biases of users and items are initilized as 0
        # the bias of rating is initilized as mean value
        self.b_s = np.zeros(self.num_src)
        self.b_t = np.zeros(self.num_tgt)
        self.b = np.mean(rating_matrix[np.where(rating_matrix != 0)])

        # Create a list of training samples (where rating > 0)
        self.samples = []
        for i in range(self.num_src):
            for j in range(self.num_tgt):
                if rating_matrix[i, j] > 0:
                    cur_tuple = [i, j, rating_matrix[i, j]]
                    src_lang = self.src_langs[i]
                    tgt_lang = self.tgt_langs[j]
                    if src_lang + "_" + tgt_lang in self.context_dict.keys():
                        cur_tuple.append(self.context_dict[src_lang + "_" + tgt_lang])
                    else:
                        raise KeyError
                    self.samples.append(tuple(cur_tuple))

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        development_process = []

        for i in range(self.iterations):
            # shuffle training samples
            if self.lr_decay != 0 and (i + 1) % 100 == 0:
                self.alpha *= (1. / (1. + self.lr_decay * (i+1)))
            np.random.shuffle(self.samples)
            self.sgd()
            
            # Calculate RMSE for train error
            xs, ys = rating_matrix.nonzero()
            error = 0
            for x, y in zip(xs, ys):
                error += pow(rating_matrix[x][y] - self.prediction[x][y], 2)
            train_rmse = np.sqrt(error / len(xs))
            training_process.append((i, train_rmse))
            
            # Eval on dev
            Y_dev_pred = self.predict(X_dev_data)
            dev_rmse = np.sqrt(mean_squared_error(Y_dev_data, Y_dev_pred))
            development_process.append((i, dev_rmse))
            
            if self.verbose:
                if (i + 1) % 100 == 0:
                    logging.debug("Iteration: %d ; rmse train error = %.4f" % (i + 1, train_rmse))
                    logging.debug("Iteration: %d ; rmse eval error = %.4f" % (i + 1, dev_rmse))

                if np.isnan(train_rmse) or np.isinf(train_rmse):
                    logging.debug('+-+-'*20)
                    logging.debug('NAN detected. Exploding or vanishing gradient. Terminate training iterations.')
                    logging.debug('+-+-' * 20)
                    break

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for sample in self.samples:
            i, j, r, z = sample[0], sample[1], sample[2], np.array(sample[3], dtype=np.float64)
            # Computer prediction and error
            prediction = self.get_rating(sample)
            self.prediction[i][j] = prediction
            e = (r - prediction)

            # Update biases
            self.b_s[i] += self.alpha * (e - self.beta_s/self.num_scores_source[i] * self.b_s[i])
            self.b_t[j] += self.alpha * (e - self.beta_t/self.num_scores_target[j] * self.b_t[j])

            # Update user and item latent feature matrices
            self.W[i, :] += self.alpha * (e * self.H[j, :] - self.beta_w/self.num_scores_source[i] * self.W[i, :])
            self.H[j, :] += self.alpha * (e * self.W[i, :] - self.beta_h/self.num_scores_target[j] * self.H[j, :])

            # Update side information parameter if necessary
            self.C += self.alpha * (e * z - self.beta_z * self.C)

    def get_rating(self, sample):
        """
        Get the predicted rating of sample
        """
        i, j, z = sample[0], sample[1], sample[3]
        prediction = self.b + self.b_s[i] + self.b_t[j] + self.W[i, :].dot(self.H[j, :].T) + self.C.dot(z.T)
        return prediction

    def predict(self, test_data):
        """
        Predict the score for testing data
        """
        pred_arrs = []
        for index, record in test_data.iterrows():
            src_lang = record["source_lang"]
            tgt_lang = record["target_lang"]
            dataset_size = str(int(record["dataset_size"]))
            src_lang_index = self.src_langs.index(f"{src_lang}_{dataset_size}")
            tgt_lang_index = self.tgt_langs.index(tgt_lang)
            cur_tuple = [src_lang_index, tgt_lang_index, 0] # Padded 0 just for the sake of get_rating
            if f"{src_lang}_{dataset_size}_{tgt_lang}" in self.context_dict.keys():
                cur_tuple.append(self.context_dict[f"{src_lang}_{dataset_size}_{tgt_lang}"])
            else:
                raise KeyError
            prediction = self.get_rating(tuple(cur_tuple))
            pred_arrs.append(prediction)
            
        return np.array(pred_arrs)

    def get_params(self, deep=True):
        return {"alpha": self.alpha, 
                "beta_w": self.beta_w,
                "beta_h": self.beta_h,
                "beta_z": self.beta_z,
                "beta_s": self.beta_s,
                "beta_t": self.beta_t
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        return self

class MFPipeline(GenericModelPipeline):
    def __init__(self, model_name, score_name, mf_json, **kwargs):
        with open(mf_json, 'r') as file:
            json_content = json.load(file)
        
        self.fixed_param = json_content.get("fixed_param", {})
        self.param_space = json_content["param_space"]
        
        # Cross validation
        self.cv = json_content["cv"]
        self.score_col_name = f"{model_name}_ft_{score_name}_mean"
        
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
        
        # Get numerical columns to be scaled
        columns_without_lang = list(X_train.columns)
        columns_without_lang.remove("source_lang")
        columns_without_lang.remove("target_lang")
        # This is [HACK] for language features need to remove train_dataset in context
        if len(columns_without_lang) < 10:
            columns_without_lang.remove("dataset_size")
        self.numerical_columns = columns_without_lang

        # Identify the numerical indices
        if 'eng_target' in self.numerical_columns:
            self.numerical_columns.remove("eng_target")

        # Fit the numerical training data
        self.scaler.fit(X_train[self.numerical_columns])

    def scale_dataset(self, X_input):
        X_input_scaled = X_input.copy()
        X_input_scaled[self.numerical_columns] = self.scaler.transform(X_input[self.numerical_columns])
        return X_input_scaled
    
    def get_context_dict_info(self, df, context_features):
        side_dict = {}
        scaled_df = self.scale_dataset(df)
        
        for index, record in df.iterrows():
            src_lang = record["source_lang"]
            tgt_lang = record["target_lang"]
            dataset_size = str(int(record["dataset_size"]))
            side_dict[f"{src_lang}_{dataset_size}_{tgt_lang}"] = scaled_df.loc[index, context_features].values
        return side_dict
    
    # Main pipeline for training the model
    def run_train(self, X_train, Y_train, score_se_df, list_X_tests, seed=RANDOM_SEED):
        def combine_src_lang_and_size(row):
            return f"{row['source_lang']}_{str(int(row['dataset_size']))}"
        
        # Get columns without language columns
        columns_without_lang = list(X_train.columns)
        columns_without_lang.remove("source_lang")
        columns_without_lang.remove("target_lang")
        # This is [HACK] for language features need to remove train_dataset in context
        if len(columns_without_lang) < 10:
            columns_without_lang.remove("dataset_size")
        
        # Create empty language matrix based on X_train_data; combine source_lang with dataset_size
        src_langs = set(X_train.apply(combine_src_lang_and_size, axis=1))
        tgt_langs = set(X_train["target_lang"].unique())
        for X_test in list_X_tests:
            src_langs.update(X_test.apply(combine_src_lang_and_size, axis=1))
            tgt_langs.update(X_test["target_lang"].unique())
        src_langs = list(src_langs)
        tgt_langs = list(tgt_langs)
        random.shuffle(src_langs)
        random.shuffle(tgt_langs)
        self.langs_matrix = pd.DataFrame(index=src_langs, columns=tgt_langs)
        
        # Before scaling, remove language columns from original columns
        self.setup_scaler(X_train)

        # Context information
        full_df = pd.concat([X_train] + list_X_tests, ignore_index=True)
        context_dict = self.get_context_dict_info(full_df, columns_without_lang)
        context_info_len = len(columns_without_lang)

        # Manual "Grid Search"; compute the Cartesian product of parameter values
        keys, values = zip(*self.param_space.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        best_param = param_combinations[0]
        best_cv_rmse = np.inf

        # Display all hyperparameter combinations
        for combination in param_combinations:
            # Initialize K-Fold Cross-Validation
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=seed)

            # Perform K-Fold Cross-Validation
            cur_rmse = 0
            for fold, (train_indices, test_indices) in enumerate(kf.split(X_train)):
                X_train_fold, X_dev_fold = X_train.iloc[train_indices], X_train.iloc[test_indices]
                Y_train_fold, Y_dev_fold = Y_train.iloc[train_indices], Y_train.iloc[test_indices]
                
                kwargs = {**self.fixed_param, **combination}
                mf_model = MFRegressor(K=20, context_dict=context_dict, context_info_len=context_info_len,
                                    random_state=seed, **kwargs)
                mf_model.fit(X_train_fold, Y_train_fold, X_dev_fold, Y_dev_fold, self.score_col_name, self.langs_matrix)
                Y_pred = mf_model.predict(X_dev_fold)
                
                # Calculate MSE of validation set
                cur_rmse += mean_squared_error(Y_pred, Y_dev_fold)
            
            # If current RMSE combination is lower, 
            cur_rmse = np.sqrt(cur_rmse / self.cv)
            if cur_rmse < best_cv_rmse:
                best_cv_rmse = cur_rmse
                best_param = combination
            
        # Get the best parameters
        self.aggregator.set_best_params(best_param)

        # Evaluate RMSE from cross-validation, taking the negative to get positive RMSE
        self.aggregator.set_cv_rmse(best_cv_rmse)
            
        # With the best hyperparameter, we retrain with original dataset, split into train-test again
        best_kwargs = {**self.fixed_param, **best_param}
        self.best_model = MFRegressor(K=20, context_dict=context_dict, context_info_len=context_info_len,
                            random_state=seed, **best_kwargs)
        X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(X_train, Y_train, test_size=0.2, random_state=seed, stratify=X_train["source_lang"])
        self.best_model.fit(X_train_split, Y_train_split, X_val_split, Y_val_split, self.score_col_name, self.langs_matrix)

    def perform_prediction(self, X_test):
        # Predict missing values
        Y_pred = self.best_model.predict(X_test)
        return Y_pred
