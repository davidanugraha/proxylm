import json

from sklearn.metrics import mean_squared_error

from .utils import *
from .regressor_models.xgboost import XGBPipeline
from .regressor_models.lgbm import LGBMPipeline
from .regressor_models.poly_regressor import PolyPipeline
from .regressor_models.mf import MFPipeline

NUM_REPEATS = 5

def init_model_pipeline(regressor_name, model_name, score_name, regressor_json_path):
    # Parse argument for model to be used
    if regressor_name == "xgb":
        # XGBoost
        return XGBPipeline(model_name, score_name, regressor_json_path)
    elif regressor_name == "lgbm":
        # LGBM
        return LGBMPipeline(model_name, score_name, regressor_json_path)
    elif regressor_name == "poly":
        # Poly (any degree depending on config)
        return PolyPipeline(model_name, score_name, regressor_json_path)
    elif regressor_name == "mf":
        # Matrix factorization
        return MFPipeline(model_name, score_name, regressor_json_path)
    else:
        # Raise error argument is not recognized
        raise ValueError(f"Unsupported regressor type: {regressor_name}")

def run_random(regressor_name, dataset_name, regressor_json_path, model_name, score_name="spBLEU"):
    logging.debug(f"Running random experiment for estimated model {model_name} on {dataset_name} with scoring {score_name}")
    
    full_random_df = pd.DataFrame()
         
    # Gather dataset; try all combinations
    feature_combinations = get_all_features_combinations()

    for feature_combo in feature_combinations:
        logging.debug(f"Running with features: {feature_combo}")
        
        # Initialize model pipeline
        model_pipeline = init_model_pipeline(regressor_name, model_name, score_name, regressor_json_path)
        
        # Hack to include language columns only for MF
        include_lang_cols = True if regressor_name == "mf" else False

        for i in range(NUM_REPEATS):
            set_seed_all(i + RANDOM_SEED)
            X_train, X_test, Y_train, Y_test, score_se_df, X_columns = get_dataset_random(model_name, dataset_name, score_name=score_name, test_size=TEST_SIZE,
                                                                  include_lang_cols=include_lang_cols,
                                                                  nlperf_only=feature_combo["nlperf_only"],
                                                                  dataset_features=feature_combo["dataset_features"],
                                                                  lang_features=feature_combo["lang_features"],
                                                                  with_trfm=feature_combo["with_trfm"],
                                                                  with_small100_ft=feature_combo["with_small100_ft"],
                                                                  with_small100_noft=feature_combo["with_small100_noft"],
                                                                  with_model_noft=feature_combo["with_model_noft"],
                                                                  seed=(i + RANDOM_SEED))
            
            # Run train
            model_pipeline.run_train(X_train, Y_train, [X_test], seed=(i + RANDOM_SEED))
            
            # Get prediction
            Y_pred = model_pipeline.perform_prediction(X_test)
            
            # Test
            test_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
            model_pipeline.aggregate_result(test_rmse, X_columns)
            
        # Retrieve aggregated result, pad experiment columns
        result_df = model_pipeline.retrieve_aggregated_result(X_columns)
        result_df["nlperf_only"] = feature_combo["nlperf_only"]
        result_df["dataset_features"] = feature_combo["dataset_features"]
        result_df["lang_features"] = feature_combo["lang_features"]
        result_df["with_trfm"] = feature_combo["with_trfm"]
        result_df["with_small100_ft"] = feature_combo["with_small100_ft"]
        result_df["with_small100_noft"] = feature_combo["with_small100_noft"]
        result_df["with_model_noft"] = feature_combo["with_model_noft"]

        full_random_df = pd.concat([full_random_df, result_df], ignore_index=True)
    
    full_random_df.to_csv(f'random_results_{score_name}_{model_name}_{dataset_name}_{regressor_name}.csv', index=False)

            
def run_lolo(regressor_name, dataset_name, regressor_json_path, model_name, score_name="spBLEU", query_lang="all"):
    logging.debug(f"Running LOLO experiment for estimated model {model_name} on {dataset_name} with scoring {score_name}")
    
    full_lolo_df = pd.DataFrame()
      
    # Gather dataset; try all combinations
    feature_combinations = get_all_features_combinations()

    if query_lang == "all":
        all_langs = get_all_languages(dataset_name=dataset_name, score_name=score_name)
    else:
        all_langs = [query_lang]
        
    for lang in all_langs:
        logging.debug(f"Running LOLO on language {lang}")
        for feature_combo in feature_combinations:
            logging.debug(f"Running with features: {feature_combo}")
            
            # Initialize model
            model_pipeline = init_model_pipeline(regressor_name, model_name, score_name, regressor_json_path)
            
            # Hack to include language columns only for MF
            include_lang_cols = True if regressor_name == "mf" else False
            
            for i in range(NUM_REPEATS):
                set_seed_all(i + RANDOM_SEED)
                X_train, X_test_source, X_test_target, Y_train, Y_test_source, Y_test_target, score_se_df, X_columns = get_dataset_lolo(model_name, dataset_name, lang=lang, score_name=score_name, 
                                                                            include_lang_cols=include_lang_cols,
                                                                            nlperf_only=feature_combo["nlperf_only"],
                                                                            dataset_features=feature_combo["dataset_features"],
                                                                            lang_features=feature_combo["lang_features"],
                                                                            with_trfm=feature_combo["with_trfm"],
                                                                            with_small100_ft=feature_combo["with_small100_ft"],
                                                                            with_small100_noft=feature_combo["with_small100_noft"],
                                                                            with_model_noft=feature_combo["with_model_noft"],
                                                                            seed=(i + RANDOM_SEED))
                
                # Run train
                model_pipeline.run_train(X_train, Y_train, [X_test_source, X_test_target], seed=(i + RANDOM_SEED))
                
                # Get prediction
                Y_pred_source = model_pipeline.perform_prediction(X_test_source)
                Y_pred_target = model_pipeline.perform_prediction(X_test_target)
                
                # Test
                test_source_rmse = np.sqrt(mean_squared_error(Y_test_source, Y_pred_source))
                test_target_rmse = np.sqrt(mean_squared_error(Y_test_target, Y_pred_target))
                model_pipeline.aggregate_result_lolo(test_source_rmse, test_target_rmse, X_columns)
                
            # Retrieve aggregated result, pad experiment columns
            result_df = model_pipeline.retrieve_aggregated_result_lolo(X_columns)
            result_df["lang"] = lang
            result_df["nlperf_only"] = feature_combo["nlperf_only"]
            result_df["dataset_features"] = feature_combo["dataset_features"]
            result_df["lang_features"] = feature_combo["lang_features"]
            result_df["with_trfm"] = feature_combo["with_trfm"]
            result_df["with_small100_ft"] = feature_combo["with_small100_ft"]
            result_df["with_small100_noft"] = feature_combo["with_small100_noft"]
            result_df["with_model_noft"] = feature_combo["with_model_noft"]
            
            full_lolo_df = pd.concat([full_lolo_df, result_df], ignore_index=True)
                
    full_lolo_df.to_csv(f'full_lolo_results_{score_name}_{model_name}_{dataset_name}_{regressor_name}_{query_lang}.csv', index=False)
    
def run_seen_unseen(regressor_name, dataset_name, regressor_json_path, model_name, score_name="spBLEU"):
    logging.debug(f"Running LOLO (Seen-Unseen) experiment for estimated model {model_name} on {dataset_name} with scoring {score_name}")
    
    full_seen_unseen_df = pd.DataFrame()
      
    # Gather dataset; try all combinations
    feature_combinations = get_all_features_combinations()

    for train_is_seen in [True, False]:
        if train_is_seen:
            test_langs = get_all_unseen_languages(model_name=model_name, dataset_name=dataset_name)
            logging.debug(f"Running seen languages for unseen languages")
        else:
            test_langs = get_all_seen_languages(model_name=model_name, dataset_name=dataset_name)
            logging.debug(f"Running unseen languages for seen languages")
            
        test_column_names = []
        for lang in test_langs:
            test_column_names.append(f"test_source_{lang}_rmse")
            test_column_names.append(f"test_target_{lang}_rmse")
        
        for feature_combo in feature_combinations:             
            logging.debug(f"Running with features: {feature_combo}")
            
            # Initialize model
            model_pipeline = init_model_pipeline(regressor_name, model_name, score_name, regressor_json_path)
            
            # Hack to include language columns only for MF
            include_lang_cols = True if regressor_name == "mf" else False
            
            for i in range(NUM_REPEATS):
                set_seed_all(i + RANDOM_SEED)
                X_train, list_X_test, Y_train, list_Y_test, score_se_df, X_columns = get_dataset_seen_unseen(model_name, dataset_name, test_langs=test_langs, score_name=score_name,
                                                                            include_lang_cols=include_lang_cols,
                                                                            nlperf_only=feature_combo["nlperf_only"],
                                                                            dataset_features=feature_combo["dataset_features"],
                                                                            lang_features=feature_combo["lang_features"],
                                                                            with_trfm=feature_combo["with_trfm"],
                                                                            with_small100_ft=feature_combo["with_small100_ft"],
                                                                            with_small100_noft=feature_combo["with_small100_noft"],
                                                                            with_model_noft=feature_combo["with_model_noft"],
                                                                            seed=(i + RANDOM_SEED))
                
                # Run train
                model_pipeline.run_train(X_train, Y_train, list_X_test, seed=(i + RANDOM_SEED))
                
                # Get prediction for each of test langs
                test_rmse_list = []
                for index, X_test in enumerate(list_X_test):
                    Y_pred = model_pipeline.perform_prediction(X_test)
                    test_rmse_list.append(np.sqrt(mean_squared_error(Y_pred, list_Y_test[index])))
                model_pipeline.aggregate_result_list(test_rmse_list, test_column_names, X_columns)
                
            # Retrieve aggregated result, pad experiment columns
            result_df = model_pipeline.retrieve_aggregated_result_list(test_column_names, X_columns)
            result_df["train_is_seen"] = train_is_seen
            result_df["nlperf_only"] = feature_combo["nlperf_only"]
            result_df["dataset_features"] = feature_combo["dataset_features"]
            result_df["lang_features"] = feature_combo["lang_features"]
            result_df["with_trfm"] = feature_combo["with_trfm"]
            result_df["with_small100_ft"] = feature_combo["with_small100_ft"]
            result_df["with_small100_noft"] = feature_combo["with_small100_noft"]
            result_df["with_model_noft"] = feature_combo["with_model_noft"]
            
            full_seen_unseen_df = pd.concat([full_seen_unseen_df, result_df], ignore_index=True)
                
    full_seen_unseen_df.to_csv(f'full_seen_unseen_results_{score_name}_{model_name}_{dataset_name}_{regressor_name}.csv', index=False)


def run_cross_dataset(regressor_name, regressor_json_path, model_name, score_name="spBLEU"):
    logging.debug(f"Running Cross Dataset experiment for estimated model {model_name} with scoring {score_name}")
    
    full_cross_dataset_df = pd.DataFrame()
      
    # Gather dataset; try all combinations
    feature_combinations = get_all_features_combinations()

    for train_mt560 in [True]: 
        for feature_combo in feature_combinations:                
            logging.debug(f"Running with features: {feature_combo}")
            
            # Initialize model
            model_pipeline = init_model_pipeline(regressor_name, model_name, score_name, regressor_json_path)
            
            # Hack to include language columns only for MF
            include_lang_cols = True if regressor_name == "mf" else False
            
            for i in range(NUM_REPEATS):
                set_seed_all(i + RANDOM_SEED)
                X_train, X_test, Y_train, Y_test, score_se_df, X_columns = get_dataset_cross_dataset(model_name, train_mt560, score_name=score_name,
                                                                            include_lang_cols=include_lang_cols,
                                                                            nlperf_only=feature_combo["nlperf_only"],
                                                                            dataset_features=feature_combo["dataset_features"],
                                                                            lang_features=feature_combo["lang_features"],
                                                                            with_trfm=feature_combo["with_trfm"],
                                                                            with_small100_ft=feature_combo["with_small100_ft"],
                                                                            with_small100_noft=feature_combo["with_small100_noft"],
                                                                            with_model_noft=feature_combo["with_model_noft"],
                                                                            seed=(i + RANDOM_SEED))
                
                # Run train
                model_pipeline.run_train(X_train, Y_train, [X_test], seed=(i + RANDOM_SEED))
                
                # Get prediction
                Y_pred = model_pipeline.perform_prediction(X_test)
                
                # Test
                test_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
                model_pipeline.aggregate_result(test_rmse, X_columns)
                
            # Retrieve aggregated result, pad experiment columns
            result_df = model_pipeline.retrieve_aggregated_result(X_columns)
            result_df["train_mt560"] = train_mt560
            result_df["nlperf_only"] = feature_combo["nlperf_only"]
            result_df["dataset_features"] = feature_combo["dataset_features"]
            result_df["lang_features"] = feature_combo["lang_features"]
            result_df["with_trfm"] = feature_combo["with_trfm"]
            result_df["with_small100_ft"] = feature_combo["with_small100_ft"]
            result_df["with_small100_noft"] = feature_combo["with_small100_noft"]
            result_df["with_model_noft"] = feature_combo["with_model_noft"]
            
            full_cross_dataset_df = pd.concat([full_cross_dataset_df, result_df], ignore_index=True)
                
    full_cross_dataset_df.to_csv(f'full_cross_dataset_results_{score_name}_{model_name}_{regressor_name}.csv', index=False)

def run_incremental(regressor_name, dataset_name, regressor_json_path, model_name, score_name="spBLEU"):
    logging.debug(f"Running Incremental experiment for estimated model {model_name} with scoring {score_name}")
    
    full_incremental_df = pd.DataFrame()
      
    for train_portion in [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 0.9]:
        # Initialize model
        logging.debug(f"Running with train portion of: {train_portion}")
        model_pipeline = init_model_pipeline(regressor_name, model_name, score_name, regressor_json_path)
        
        for i in range(NUM_REPEATS):
            set_seed_all(i + RANDOM_SEED)
            X_train, X_test, Y_train, Y_test, score_se_df, X_columns = get_incremental_dataset(model_name, dataset_name, train_portion,
                                                                                                score_name=score_name,
                                                                                                seed=(i + RANDOM_SEED))
            
            # Run train
            model_pipeline.run_train(X_train, Y_train, [X_test], seed=(i + RANDOM_SEED))
            
            # Get prediction
            Y_pred = model_pipeline.perform_prediction(X_test)
            
            # Test
            test_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
            model_pipeline.aggregate_result(test_rmse, X_columns)
            
        # Retrieve aggregated result, pad experiment columns
        result_df = model_pipeline.retrieve_aggregated_result(X_columns)
        result_df["train_portion"] = train_portion
        result_df["train_regressor_size"] = len(X_train) 
        
        full_incremental_df = pd.concat([full_incremental_df, result_df], ignore_index=True)
                
    full_incremental_df.to_csv(f'full_incremental_results_{score_name}_{model_name}_{regressor_name}.csv', index=False)


def run_mt(exp_mode, dataset_name, regressor_name, regressor_json_path, model_name, score_name, lang="all"):
    with open(regressor_json_path, 'r') as file:
        json_content = json.load(file)
        logging.debug("Regressor json information:")
        logging.debug(json_content)
    
    # Parse argument for experiment mode
    if exp_mode == "random":
        run_random(regressor_name, dataset_name, regressor_json_path, model_name, score_name=score_name)
    elif exp_mode == "lolo":    
        run_lolo(regressor_name, dataset_name, regressor_json_path, model_name, score_name=score_name, query_lang=lang)
    elif exp_mode == "seen_unseen":
        run_seen_unseen(regressor_name, dataset_name, regressor_json_path, model_name, score_name=score_name)
    elif exp_mode == "cross_dataset":
        run_cross_dataset(regressor_name, regressor_json_path, model_name, score_name=score_name)
    elif exp_mode == "incremental":
        run_incremental(regressor_name, dataset_name, regressor_json_path, model_name, score_name=score_name)
    else:
        # Raise error argument is not recognized
        raise ValueError(f"Unknown experiment mode: {exp_mode}") 
