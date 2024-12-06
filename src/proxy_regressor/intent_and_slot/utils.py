import logging
import os
import random

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

logging.basicConfig(level=logging.DEBUG)

RANDOM_SEED = 42
N_JOBS = 32
TEST_SIZE = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

INTENT_DATA_PATH = os.path.join(CUR_DIR, "csv_datasets", "final_intent_scores.csv")
SLOT_DATA_PATH = os.path.join(CUR_DIR, "csv_datasets", "final_slot_scores.csv")
MODEL_NAMES = ["aya", "llama3", "smollm_135m", "smollm_360m", "bloomz_560m"]

# Fix for skopt error np.int deprecated
np.int = int

LANG_FEATURES = ["genetic", "geographic", "syntactic", "inventory", "phonological", "featural"]

def set_seed_all(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_all_languages(dataset_name):
    if dataset_name == "intent":
        df = get_dataframe_intent()
        all_langs = df["source_lang"].unique()
        all_langs = all_langs.tolist()
    else:
        df = get_dataframe_slot()
        all_langs = df["source_lang"].unique()
        all_langs = all_langs.tolist()
    return all_langs

def get_dataframe_intent(remove_useless_columns=True):
    df = pd.read_csv(INTENT_DATA_PATH)

    # Dropna 
    df = df.dropna()
    
    # Remove columns where all values are the same
    if remove_useless_columns:
        columns_to_drop = df.columns[df.nunique() == 1]
        df = df.drop(columns=columns_to_drop)

    return df

def get_dataframe_slot(remove_useless_columns=True):
    df = pd.read_csv(SLOT_DATA_PATH)

    # Dropna 
    df = df.dropna()
    
    # Remove columns where all values are the same
    if remove_useless_columns:
        columns_to_drop = df.columns[df.nunique() == 1]
        df = df.drop(columns=columns_to_drop)

    return df

def get_all_features_combinations():
    combs = []
    
    # All
    combs.append({"nlperf_only": False,
                  "dataset_features": True,
                  "lang_features": True,
                  "with_smollm_135m": True,
                  "with_smollm_360m": True,
                  "with_bloomz_560m": True})
    
    # NLPerf features only
    # combs.append({"nlperf_only": True,
    #               "dataset_features": False,
    #               "lang_features": False,
    #               "with_smollm_135m": False,
    #               "with_smollm_360m": False,
    #               "with_bloomz_560m": False})
    
    # proxy_models_options = [
    #     {"with_smollm_135m": False,
    #               "with_smollm_360m": False,
    #               "with_bloomz_560m": False},
    # ]
    
    # for proxy_comb in proxy_models_options:
    #     # Dataset and one of proxy models
    #     dataset_only = {"nlperf_only": False, "dataset_features": True, "lang_features": False}
    #     combs.append({**dataset_only, **proxy_comb})
        
    #     # Language and one of proxy models
    #     lang_only = {"nlperf_only": False, "dataset_features": False, "lang_features": True}
    #     combs.append({**lang_only, **proxy_comb})
        
    #     both_dataset_lang = {"nlperf_only": False, "dataset_features": True, "lang_features": True}
    #     combs.append({**both_dataset_lang, **proxy_comb})

    # Define options for proxy models
    # proxy_models_options = [
    #     {"with_smollm_135m": True, "with_smollm_360m": False, "with_bloomz_560m": False},
    #     {"with_smollm_135m": False, "with_smollm_360m": True, "with_bloomz_560m": False},
    #     {"with_smollm_135m": False, "with_smollm_360m": False, "with_bloomz_560m": True},
        # {"with_smollm_135m": False, "with_smollm_360m": False, "with_bloomz_560m": False},
    # ]
    
    
    # for proxy_comb in proxy_models_options:
        # # Dataset and one of proxy models
        # dataset_only = {"nlperf_only": False, "dataset_features": True, "lang_features": False}
        # combs.append({**dataset_only, **proxy_comb})
        
        # # Language and one of proxy models
        # lang_only = {"nlperf_only": False, "dataset_features": False, "lang_features": True}
        # combs.append({**lang_only, **proxy_comb})
        
        # Dataset and language and one of proxy models
        # both_dataset_lang = {"nlperf_only": False, "dataset_features": True, "lang_features": True}
        # combs.append({**both_dataset_lang, **proxy_comb})
    
    return combs
    
def select_features(df, model_name, score_name, include_lang_cols=False, nlperf_only=False,
                    dataset_features=True, lang_features=True,
                    with_smollm_135m=True, with_smollm_360m=True,
                    with_bloomz_560m=True):
    # Select features based on dictionary option
    if nlperf_only:
        # NLPerf features
        columns_features = ["genetic", "geographic", "syntactic", "inventory", "phonological", "featural",
                            "train_word_vocab_size", "test_word_vocab_size",
                            "avg_train_sentence_length", "avg_test_sentence_length",
                            "train_test_word_overlap", "ttr_train",
                            "ttr_test","train_test_ttr_distance"]
        # For columns that got filtered out for Nusa dataset
        remove_cols = []
        for col in columns_features:
            if col not in df.columns:
                remove_cols.append(col)
        
        for col in remove_cols:
            columns_features.remove(col)
            
    else:
        columns_features = []
        score_columns = [f"{score_col}_{score_name}" for score_col in MODEL_NAMES]

        # Our hand-engineered features
        non_numerical_features = ['source_lang', 'target_lang', 'train_dataset', 'test_dataset']
        for col in df.columns:
            add_column = True
            for score_col in score_columns:
                if score_col in col or col in non_numerical_features:
                    add_column = False
                    break
            
            if add_column:
                if lang_features and col in LANG_FEATURES:
                    # lang feats is enabled and current col is lang features
                    columns_features.append(col)
                elif dataset_features and col not in LANG_FEATURES:
                    # col is not lang features, and dataset features is enabled
                    columns_features.append(col)
        
        if with_smollm_135m:
            columns_features.append(f"smollm_135m_{score_name}")
        
        if with_smollm_360m:
            columns_features.append(f"smollm_360m_{score_name}")            

        if with_bloomz_560m:
            columns_features.append(f"bloomz_560m_{score_name}")
            
    if include_lang_cols:
        columns_features.insert(0, "target_lang")
        columns_features.insert(0, "source_lang")
            
    return columns_features

def get_dataset_random(model_name, dataset_name, score_name="accuracy", test_size=TEST_SIZE,
                       include_lang_cols=False, remove_useless_columns=True,
                       nlperf_only=False, dataset_features=True, lang_features=True,
                       with_smollm_135m=True, with_smollm_360m=True,
                       with_bloomz_560m=True, seed=RANDOM_SEED):
    # Get dataframe and select appropriate features
    if dataset_name == "intent":
        df = get_dataframe_intent(remove_useless_columns=remove_useless_columns)
    else:
        df = get_dataframe_slot(remove_useless_columns=remove_useless_columns)
    
    X_features = select_features(df, model_name=model_name, score_name=score_name, include_lang_cols=include_lang_cols,
                                 nlperf_only=nlperf_only, dataset_features=dataset_features,
                                 lang_features=lang_features,
                                 with_smollm_135m=with_smollm_135m, with_smollm_360m=with_smollm_360m,
                                 with_bloomz_560m=with_bloomz_560m)
    
    Y = df[f'{model_name}_{score_name}']

    X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=test_size, random_state=seed, stratify=df['source_lang'])
    X_train = X_train[X_features]
    X_test = X_test[X_features]
    
    return X_train, X_test, Y_train, Y_test, X_features

def get_dataset_lolo(model_name, dataset_name, lang, score_name="accuracy",
                     include_lang_cols=False, remove_useless_columns=True,
                     nlperf_only=False, dataset_features=True, lang_features=True,
                     with_smollm_135m=True, with_smollm_360m=True,
                     with_bloomz_560m=True,  seed=RANDOM_SEED):
    # Get dataframe and select appropriate features
    if dataset_name == "intent":
        df = get_dataframe_intent(remove_useless_columns=remove_useless_columns)
    else:
        df = get_dataframe_slot(remove_useless_columns=remove_useless_columns)
    
    X_features = select_features(df, model_name=model_name, score_name=score_name, include_lang_cols=include_lang_cols,
                                 nlperf_only=nlperf_only, dataset_features=dataset_features,
                                 lang_features=lang_features,
                                 with_smollm_135m=with_smollm_135m, with_smollm_360m=with_smollm_360m,
                                 with_bloomz_560m=with_bloomz_560m)
    

    df_train = df[(df["source_lang"] != lang) & (df["target_lang"] != lang)]
    df_test_source = df[df["source_lang"] == lang]
    df_test_target = df[df["target_lang"] == lang]
    
    # Shuffle then filter
    Y_train = df_train[f'{model_name}_{score_name}']
    X_train, Y_train = shuffle(df_train, Y_train, random_state=seed)

    X_train = X_train[X_features]
    X_test_source = df_test_source[X_features]
    X_test_target = df_test_target[X_features]
    Y_test_source = df_test_source[f'{model_name}_{score_name}']
    Y_test_target = df_test_target[f'{model_name}_{score_name}']
    
    return X_train, X_test_source, X_test_target, Y_train, Y_test_source, \
           Y_test_target, X_train.columns
