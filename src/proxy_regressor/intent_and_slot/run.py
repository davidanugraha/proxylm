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

def run_random(regressor_name, dataset_name, regressor_json_path, model_name, score_name="accuracy"):
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
            X_train, X_test, Y_train, Y_test, X_columns = get_dataset_random(model_name, dataset_name, score_name=score_name, test_size=TEST_SIZE,
                                                                  include_lang_cols=include_lang_cols,
                                                                  nlperf_only=feature_combo["nlperf_only"],
                                                                  dataset_features=feature_combo["dataset_features"],
                                                                  lang_features=feature_combo["lang_features"],
                                                                  with_smollm_135m=feature_combo["with_smollm_135m"],
                                                                  with_smollm_360m=feature_combo["with_smollm_360m"],
                                                                  with_bloomz_560m=feature_combo["with_bloomz_560m"],
                                                                  seed=(i + RANDOM_SEED))
            
            # Run train
            model_pipeline.run_train(X_train, Y_train, None, [X_test], seed=(i + RANDOM_SEED))
            
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
        result_df["with_smollm_135m"] = feature_combo["with_smollm_135m"]
        result_df["with_smollm_360m"] = feature_combo["with_smollm_360m"]
        result_df["with_bloomz_560m"] = feature_combo["with_bloomz_560m"]

        full_random_df = pd.concat([full_random_df, result_df], ignore_index=True)
    
    full_random_df.to_csv(f'random_results_{score_name}_{model_name}_{dataset_name}_{regressor_name}.csv', index=False)

            
def run_lolo(regressor_name, dataset_name, regressor_json_path, model_name, score_name="accuracy", query_lang="all"):
    logging.debug(f"Running LOLO experiment for estimated model {model_name} on {dataset_name} with scoring {score_name}")
    
    full_lolo_df = pd.DataFrame()
      
    # Gather dataset; try all combinations
    feature_combinations = get_all_features_combinations()

    if query_lang == "all":
        all_langs = get_all_languages(dataset_name=dataset_name)
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
                X_train, X_test_source, X_test_target, Y_train, Y_test_source, Y_test_target, X_columns = get_dataset_lolo(model_name, dataset_name, lang=lang, score_name=score_name, 
                                                                            include_lang_cols=include_lang_cols,
                                                                            nlperf_only=feature_combo["nlperf_only"],
                                                                            dataset_features=feature_combo["dataset_features"],
                                                                            lang_features=feature_combo["lang_features"],
                                                                            with_smollm_135m=feature_combo["with_smollm_135m"],
                                                                            with_smollm_360m=feature_combo["with_smollm_360m"],
                                                                            with_bloomz_560m=feature_combo["with_bloomz_560m"],
                                                                            seed=(i + RANDOM_SEED))
                
                
                list_X_tests = [X_test_source, X_test_target]
                
                # Run train
                model_pipeline.run_train(X_train, Y_train, None, list_X_tests, seed=(i + RANDOM_SEED))
                
                # Get prediction
                Y_pred_source = model_pipeline.perform_prediction(X_test_source)
                test_source_rmse = np.sqrt(mean_squared_error(Y_test_source, Y_pred_source))
                if len(X_test_target) != 0:
                    Y_pred_target = model_pipeline.perform_prediction(X_test_target)
                    test_target_rmse = np.sqrt(mean_squared_error(Y_test_target, Y_pred_target))
                    model_pipeline.aggregate_result_lolo(test_source_rmse, test_target_rmse, X_columns)
                else:
                    model_pipeline.aggregate_result_lolo_source_only(test_source_rmse, X_columns)
                    
            # Retrieve aggregated result, pad experiment columns
            result_df = model_pipeline.retrieve_aggregated_result_lolo(X_columns)
            result_df["lang"] = lang
            result_df["nlperf_only"] = feature_combo["nlperf_only"]
            result_df["dataset_features"] = feature_combo["dataset_features"]
            result_df["lang_features"] = feature_combo["lang_features"]
            result_df["with_smollm_135m"] = feature_combo["with_smollm_135m"]
            result_df["with_smollm_360m"] = feature_combo["with_smollm_360m"]
            result_df["with_bloomz_560m"] = feature_combo["with_bloomz_560m"]
            
            full_lolo_df = pd.concat([full_lolo_df, result_df], ignore_index=True)
                
    full_lolo_df.to_csv(f'full_lolo_results_{score_name}_{model_name}_{dataset_name}_{regressor_name}_{query_lang}.csv', index=False)

def run_intent_or_slot(exp_mode, dataset_name, regressor_name, regressor_json_path, model_name, score_name, lang="all"):
    with open(regressor_json_path, 'r') as file:
        json_content = json.load(file)
        logging.debug("Regressor json information:")
        logging.debug(json_content)
    
    # Parse argument for experiment mode
    if exp_mode == "random":
        run_random(regressor_name, dataset_name, regressor_json_path, model_name, score_name=score_name)
    elif exp_mode == "lolo":    
        run_lolo(regressor_name, dataset_name, regressor_json_path, model_name, score_name=score_name, query_lang=lang)
    else:
        # Raise error argument is not recognized
        raise ValueError(f"Unknown experiment mode: {exp_mode}") 
