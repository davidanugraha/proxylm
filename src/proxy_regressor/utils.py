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
N_JOBS = -1
TEST_SIZE = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

MT560_DATA_PATH = os.path.join(CUR_DIR, "csv_datasets", "mt560_experiments.csv")
NUSA_DATA_PATH = os.path.join(CUR_DIR, "csv_datasets", "nusa_experiments.csv")

# Fix for skopt error np.int deprecated
np.int = int

SCORE_COLUMNS = ['spBLEU_mean', 'spBLEU_se', 'chrF2++_mean', 'chrF2++_se',
                 'BLEU_mean', 'BLEU_se', 'chrF2_mean', 'chrF2_se',
                 'TER_mean', 'TER_se', 'comet_score_mean', 'comet_score_se']
LANG_FEATURES = ["genetic", "geographic", "syntactic", "inventory", "phonological", "featural"]

M2M100_UNSEEN_LANGUAGES = ['asm', 'dik', 'ewe', 'fao', 'hne', 'kab', 'kin', 'kir', 'lmo', 'mri',
                           'sna', 'tat', 'tel', 'tgk', 'tuk', 'uig',
                           'abs', 'btk', 'bew', 'bhp', 'mad', 'mak', 'min']
NLLB_UNSEEN_LANGUAGES = ['abs', 'btk', 'bew', 'bhp', 'mad', 'mak']

def set_seed_all(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_all_languages(dataset_name):
    if dataset_name == "mt560":
        df = get_dataframe_mt560()
        all_langs = df["source_lang"].unique()
        all_langs = all_langs.tolist()
        all_langs.remove("eng")
        return all_langs
    else:
        df = get_dataframe_nusa(remove_useless_columns=True)
        all_langs = df["source_lang"].unique()
        all_langs = all_langs.tolist()
        return all_langs

def get_all_seen_languages(model_name, dataset_name):
    all_langs = get_all_languages(dataset_name)
    if model_name == "m2m100":
        for unseen_lang in M2M100_UNSEEN_LANGUAGES:
            if unseen_lang in all_langs:
                all_langs.remove(unseen_lang)
    elif model_name == "nllb":
        for unseen_lang in NLLB_UNSEEN_LANGUAGES:
            if unseen_lang in all_langs:
                all_langs.remove(unseen_lang)
    return all_langs

def get_all_unseen_languages(model_name, dataset_name):
    if model_name == "m2m100":
        all_langs = get_all_languages(dataset_name)
        unseen_langs_list = []
        for lang in M2M100_UNSEEN_LANGUAGES:
            if lang in all_langs:
                unseen_langs_list.append(lang)
        return unseen_langs_list
    elif model_name == "nllb":
        all_langs = get_all_languages(dataset_name)
        unseen_langs_list = []
        for lang in NLLB_UNSEEN_LANGUAGES:
            if lang in all_langs:
                unseen_langs_list.append(lang)
        return unseen_langs_list

def get_dataframe_mt560():
    df = pd.read_csv(MT560_DATA_PATH)

    # Dropna 
    df = df.dropna()

    return df

def get_dataframe_nusa(remove_useless_columns=True):
    df = pd.read_csv(NUSA_DATA_PATH)

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
                  "with_trfm": True,
                  "with_small100_ft": True,
                  "with_small100_noft": True,
                  "with_model_noft": True})
    
    # NLPerf features only
    combs.append({"nlperf_only": True,
                  "dataset_features": False,
                  "lang_features": False,
                  "with_trfm": False,
                  "with_small100_ft": False,
                  "with_small100_noft": False,
                  "with_model_noft": False})

    # Define options for proxy models
    proxy_models_options = [
        {"with_trfm": False, "with_small100_ft": False, "with_small100_noft": False, "with_model_noft": False},
        {"with_trfm": True, "with_small100_ft": False, "with_small100_noft": False, "with_model_noft": False},
        {"with_trfm": False, "with_small100_ft": True, "with_small100_noft": False, "with_model_noft": False},
        {"with_trfm": False, "with_small100_ft": False, "with_small100_noft": True, "with_model_noft": False},
        {"with_trfm": False, "with_small100_ft": True, "with_small100_noft": True, "with_model_noft": False},
        {"with_trfm": False, "with_small100_ft": False, "with_small100_noft": False, "with_model_noft": True}
    ]
    
    
    for proxy_comb in proxy_models_options:
        # Dataset and one of proxy models
        dataset_only = {"nlperf_only": False, "dataset_features": True, "lang_features": False}
        combs.append({**dataset_only, **proxy_comb})
        
        # Language and one of proxy models
        lang_only = {"nlperf_only": False, "dataset_features": False, "lang_features": True}
        combs.append({**lang_only, **proxy_comb})
        
        # Dataset and language and one of proxy models
        both_dataset_lang = {"nlperf_only": False, "dataset_features": True, "lang_features": True}
        combs.append({**both_dataset_lang, **proxy_comb})
    
    return combs
    
def select_features(df, model_name, score_name="spBLEU", include_lang_cols=False, nlperf_only=False,
                    dataset_features=True, lang_features=True,
                    with_trfm=True, with_small100_ft=True,
                    with_small100_noft=True, with_model_noft=True,
                    disable_eng_target=False):
    # Select features based on dictionary option
    if nlperf_only:
        # NLPerf features
        columns_features = ["genetic", "geographic", "syntactic", "inventory", "phonological", "featural",
                            "dataset_size", "train_word_vocab_size", "dev_word_vocab_size", "test_word_vocab_size",
                            "avg_train_sentence_length", "avg_dev_sentence_length", "avg_test_sentence_length",
                            "train_dev_word_overlap", "train_test_word_overlap", "dev_test_word_overlap", "ttr_train",
                            "ttr_dev", "ttr_test", "train_dev_ttr_distance", "train_test_ttr_distance", "dev_test_ttr_distance"]
        # For columns that got filtered out for Nusa dataset
        remove_cols = []
        for col in columns_features:
            if col not in df.columns:
                remove_cols.append(col)
        
        for col in remove_cols:
            columns_features.remove(col)
    else:
        columns_features = []

        # Our hand-engineered features
        non_numerical_features = ['source_lang', 'target_lang', 'train_dataset', 'test_dataset']
        for col in df.columns:
            add_column = True
            for score_col in SCORE_COLUMNS:
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

        if with_trfm:
            columns_features.append(f"trfm_{score_name}_mean")
        
        if with_small100_noft:
            columns_features.append(f"small100_noft_{score_name}_mean")
        
        if with_small100_ft:
            columns_features.append(f"small100_ft_{score_name}_mean")
        
        if with_model_noft:
            columns_features.append(f"{model_name}_noft_{score_name}_mean")
            
    if disable_eng_target and "eng_target" in columns_features:
        columns_features.remove("eng_target")
        
    if include_lang_cols:
        if "dataset_size" not in columns_features:
            columns_features.insert(0, "dataset_size")
        columns_features.insert(0, "target_lang")
        columns_features.insert(0, "source_lang")
            
    return columns_features

def get_se_score_features(model_name, score_name="spBLEU",
                          with_trfm=True, with_small100_ft=True, with_small100_noft=True, with_model_noft=True):
    se_score_features = [f"{model_name}_ft_{score_name}_se"]
    if with_trfm:
        se_score_features.append(f"trfm_{score_name}_se")
    if with_small100_ft:
        se_score_features.append(f"small100_ft_{score_name}_se")
    if with_small100_noft:
        se_score_features.append(f"small100_noft_{score_name}_se")
    if with_model_noft:
        se_score_features.append(f"{model_name}_noft_{score_name}_se")
        
    return se_score_features

def get_dataset_random(model_name, dataset_name, score_name="spBLEU", test_size=TEST_SIZE,
                       include_lang_cols=False, remove_useless_columns=True,
                       nlperf_only=False, dataset_features=True, lang_features=True, with_trfm=True, with_small100_ft=True, with_small100_noft=True,
                       with_model_noft=True, seed=RANDOM_SEED):
    # Get dataframe and select appropriate features
    if dataset_name == "mt560":
        df = get_dataframe_mt560()
    else:
        df = get_dataframe_nusa(remove_useless_columns=remove_useless_columns)
    
    X_features = select_features(df, model_name=model_name, score_name=score_name, include_lang_cols=include_lang_cols,
                                 nlperf_only=nlperf_only, dataset_features=dataset_features,
                                 lang_features=lang_features,with_trfm=with_trfm,
                                 with_small100_ft=with_small100_ft, with_small100_noft=with_small100_noft,
                                 with_model_noft=with_model_noft)
    se_score_features = get_se_score_features(model_name, score_name=score_name, with_trfm=with_trfm,
                                 with_small100_ft=with_small100_ft, with_small100_noft=with_small100_noft,
                                 with_model_noft=with_model_noft)
    
    Y = df[f'{model_name}_ft_{score_name}_mean']

    if dataset_name == "mt560":
        df["same_domain"] = df["train_dataset"] == df["test_dataset"]
        X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=test_size, random_state=seed, stratify=df["same_domain"])
        X_train = X_train.drop("same_domain", axis=1)
        X_test = X_test.drop("same_domain", axis=1)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=test_size, random_state=seed, stratify=df["source_lang"])
    
    score_se_df = X_train[se_score_features]
    X_train = X_train[X_features]
    X_test = X_test[X_features]
    
    return X_train, X_test, Y_train, Y_test, score_se_df, X_features

def get_dataset_lolo(model_name, dataset_name, lang, score_name="spBLEU",
                     include_lang_cols=False, remove_useless_columns=True,
                     nlperf_only=False, dataset_features=True, lang_features=True, with_trfm=True, with_small100_ft=True,
                     with_small100_noft=True, with_model_noft=True, seed=RANDOM_SEED):
    # Get dataframe and select appropriate features
    if dataset_name == "mt560":
        df = get_dataframe_mt560()
    else:
        df = get_dataframe_nusa(remove_useless_columns=remove_useless_columns)
    
    X_features = select_features(df, model_name=model_name, score_name=score_name, include_lang_cols=include_lang_cols,
                                 nlperf_only=nlperf_only, dataset_features=dataset_features,
                                 lang_features=lang_features,with_trfm=with_trfm,
                                 with_small100_ft=with_small100_ft, with_small100_noft=with_small100_noft,
                                 with_model_noft=with_model_noft)
    se_score_features = get_se_score_features(model_name, score_name=score_name, with_trfm=with_trfm,
                                 with_small100_ft=with_small100_ft, with_small100_noft=with_small100_noft,
                                 with_model_noft=with_model_noft)

    df_train = df[(df["source_lang"] != lang) & (df["target_lang"] != lang)]
    df_test_source = df[df["source_lang"] == lang]
    df_test_target = df[df["target_lang"] == lang]
    
    # Shuffle then filter
    Y_train = df_train[f'{model_name}_ft_{score_name}_mean']
    X_train, Y_train = shuffle(df_train, Y_train, random_state=seed)
    score_se_df = X_train[se_score_features]
    X_train = X_train[X_features]
    X_test_source = df_test_source[X_features]
    X_test_target = df_test_target[X_features]
    Y_test_source = df_test_source[f'{model_name}_ft_{score_name}_mean']
    Y_test_target = df_test_target[f'{model_name}_ft_{score_name}_mean']
    
    return X_train, X_test_source, X_test_target, Y_train, Y_test_source, \
           Y_test_target, score_se_df, X_train.columns

def get_dataset_seen_unseen(model_name, dataset_name, test_langs, score_name="spBLEU",
                            include_lang_cols=False, remove_useless_columns=True,
                            nlperf_only=False, dataset_features=True, lang_features=True, with_trfm=True, with_small100_ft=True,
                            with_small100_noft=True, with_model_noft=True, seed=RANDOM_SEED):
    # Get dataframe and select appropriate features
    if dataset_name == "mt560":
        df = get_dataframe_mt560()
    else:
        df = get_dataframe_nusa(remove_useless_columns=remove_useless_columns)

    X_features = select_features(df, model_name=model_name, score_name=score_name, include_lang_cols=include_lang_cols,
                                 nlperf_only=nlperf_only, dataset_features=dataset_features,
                                 lang_features=lang_features,with_trfm=with_trfm,
                                 with_small100_ft=with_small100_ft, with_small100_noft=with_small100_noft,
                                 with_model_noft=with_model_noft)
    se_score_features = get_se_score_features(model_name, score_name=score_name, with_trfm=with_trfm,
                                 with_small100_ft=with_small100_ft, with_small100_noft=with_small100_noft,
                                 with_model_noft=with_model_noft)

    # Shuffle, then filter for X_train and Y_train
    df_train = df[~(df["source_lang"].isin(test_langs) | df["target_lang"].isin(test_langs))]
    Y_train = df_train[f'{model_name}_ft_{score_name}_mean']
    X_train, Y_train = shuffle(df_train, Y_train, random_state=seed)
    score_se_df = X_train[se_score_features]
    X_train = X_train[X_features]
    
    # Filter for list of X_test and Y_test
    list_X_test = []
    list_Y_test = []
    for lang in test_langs:
        df_test_source = df[df["source_lang"] == lang]
        df_test_target = df[df["target_lang"] == lang]
        list_X_test.append(df_test_source[X_features])
        list_X_test.append(df_test_target[X_features])
        list_Y_test.append(df_test_source[f'{model_name}_ft_{score_name}_mean'])
        list_Y_test.append(df_test_target[f'{model_name}_ft_{score_name}_mean'])
    
    return X_train, list_X_test, Y_train, list_Y_test, score_se_df, X_train.columns


def get_dataset_cross_dataset(model_name, train_mt560, score_name="spBLEU",
                              include_lang_cols=False,
                              nlperf_only=False, dataset_features=True, lang_features=True, with_trfm=True, with_small100_ft=True, with_small100_noft=True,
                              with_model_noft=True, seed=RANDOM_SEED):
    # Get dataset from corresponding train-test split
    if train_mt560:
        df_train = get_dataframe_mt560()
        df_test = get_dataframe_nusa(remove_useless_columns=False)
    else:
        df_train = get_dataframe_nusa(remove_useless_columns=False)
        df_test = get_dataframe_mt560()
    
    X_features = select_features(df_train, model_name=model_name, score_name=score_name, include_lang_cols=include_lang_cols,
                                 nlperf_only=nlperf_only, dataset_features=dataset_features,
                                 lang_features=lang_features,with_trfm=with_trfm,
                                 with_small100_ft=with_small100_ft, with_small100_noft=with_small100_noft,
                                 with_model_noft=with_model_noft, disable_eng_target=True)
    se_score_features = get_se_score_features(model_name, score_name=score_name, with_trfm=with_trfm,
                                 with_small100_ft=with_small100_ft, with_small100_noft=with_small100_noft,
                                 with_model_noft=with_model_noft)

    # Shuffle then filter
    Y_train = df_train[f'{model_name}_ft_{score_name}_mean']
    X_train, Y_train = shuffle(df_train, Y_train, random_state=seed)
    score_se_df = X_train[se_score_features]
    X_train = X_train[X_features]
    X_test = df_test[X_features]
    Y_test = df_test[f'{model_name}_ft_{score_name}_mean']

    return X_train, X_test, Y_train, Y_test, score_se_df, X_train.columns

def get_incremental_dataset(model_name, dataset_name, train_portion, score_name="spBLEU", test_size=TEST_SIZE,
                            remove_useless_columns=True, seed=RANDOM_SEED):
    # Get dataframe and select appropriate features
    if dataset_name == "mt560":
        df = get_dataframe_mt560()
    else:
        df = get_dataframe_nusa(remove_useless_columns=remove_useless_columns)
    
    # Use best setting only
    X_features = select_features(df, model_name=model_name, score_name=score_name,
                                 nlperf_only=False, dataset_features=True,
                                 lang_features=True,with_trfm=True,
                                 with_small100_ft=True, with_small100_noft=True,
                                 with_model_noft=True)
    se_score_features = get_se_score_features(model_name, score_name=score_name, with_trfm=True,
                                 with_small100_ft=True, with_small100_noft=True,
                                 with_model_noft=True)
    
    Y = df[f'{model_name}_ft_{score_name}_mean']

    if dataset_name == "mt560":
        df["same_domain"] = df["train_dataset"] == df["test_dataset"]
        X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=test_size, random_state=seed, stratify=df["same_domain"])
        X_train = X_train.drop("same_domain", axis=1)
        X_test = X_test.drop("same_domain", axis=1)
        
        # Cut portion based on train portion
        X_train_copy = X_train.copy(True)
        X_train_copy["same_domain"] = X_train_copy["train_dataset"] == X_train_copy["test_dataset"]
        _, X_train, _, Y_train = train_test_split(X_train_copy, Y_train, test_size=train_portion,
                                                  random_state=seed, stratify=X_train_copy["same_domain"])
        X_train = X_train.drop("same_domain", axis=1)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=test_size, random_state=seed, stratify=df["source_lang"])
        
        # Cut portion based on train portion
        X_train_copy = X_train.copy(True)
        _, X_train, _, Y_train = train_test_split(X_train_copy, Y_train, test_size=train_portion,
                                                  random_state=seed, stratify=X_train_copy["source_lang"])
    
    score_se_df = X_train[se_score_features]
    X_train = X_train[X_features]
    X_test = X_test[X_features]
    
    return X_train, X_test, Y_train, Y_test, score_se_df, X_features
    