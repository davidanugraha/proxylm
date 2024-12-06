import yaml
import argparse
import logging
import os

from .intent_and_slot.run import run_intent_or_slot
from .mt.run import run_mt

logging.basicConfig(level=logging.DEBUG)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CUR_DIR))

VALID_DATASET_NAME = ['mt560', 'nusa', 'intent', 'slot']
VALID_REGRESSOR_NAME = ['xgb', 'lgbm', 'mf', 'poly']

VALID_MT_MODELS = ['m2m100', 'nllb']
VALID_MT_EXP_MODE = ['random', 'lolo', 'unseen', 'cross_dataset', 'incremental']
VALID_INTENT_SLOT_EXP_MODE = ['random', 'lolo']
VALID_INTENT_SLOT_MODELS = ['aya', 'llama3']

def load_yaml_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error reading YAML config: {exc}")
            return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_yaml_config(args.config)
    if not config:
        raise ValueError("Failed to load configuration. Exiting.")

    # Validate and access configuration parameters
    regressor_json = config.get('regressor_config')
    if regressor_json and not os.path.isabs(regressor_json):
        regressor_json = os.path.join(ROOT_DIR, regressor_json)
        
    regressor = config.get('regressor')
    if regressor not in VALID_REGRESSOR_NAME:
        raise NotImplementedError(f"Regressor `{regressor}` is not recognized! Valid options are: {VALID_REGRESSOR_NAME}")
    
    dataset_name = config.get('dataset_name')
    if dataset_name == "mt560":
        if regressor == "mf":
            raise ValueError(f"Matrix factorization (mf) cannot be used for MT560 dataset")
        score = config.get('score', 'spBLEU')

        model = config.get('model')
        if model not in VALID_MT_MODELS:
            raise ValueError(f"Model `{model}` is not recognized! Valid options are: {VALID_MT_MODELS}")
        
        exp_mode = config.get('exp_mode')
        if exp_mode not in VALID_MT_EXP_MODE:
            raise ValueError(f"Experiment mode `{exp_mode}` is not recognized! Valid options are: {VALID_MT_EXP_MODE}")
        
    elif dataset_name == "nusa":
        score = config.get('score', 'spBLEU')
        
        model = config.get('model')
        if model not in VALID_MT_MODELS:
            raise ValueError(f"Model `{model}` is not recognized! Valid options are: {VALID_MT_MODELS}")
        
        exp_mode = config.get('exp_mode')
        if exp_mode not in VALID_MT_EXP_MODE:
            raise ValueError(f"Experiment mode `{exp_mode}` is not recognized! Valid options are: {VALID_MT_EXP_MODE}")
        
    elif dataset_name == "intent":
        score = config.get('score', 'accuracy')
        
        model = config.get('model')
        if model not in VALID_INTENT_SLOT_MODELS:
            raise ValueError(f"Model `{model}` is not recognized! Valid options are: {VALID_INTENT_SLOT_MODELS}")
        
        exp_mode = config.get('exp_mode')
        if exp_mode not in VALID_INTENT_SLOT_EXP_MODE:
            raise ValueError(f"Experiment mode `{exp_mode}` is not recognized! Valid options are: {VALID_INTENT_SLOT_EXP_MODE}")
    elif dataset_name == "slot":
        score = config.get('score', 'f1')
        
        model = config.get('model')
        if model not in VALID_INTENT_SLOT_MODELS:
            raise ValueError(f"Model `{model}` is not recognized! Valid options are: {VALID_INTENT_SLOT_MODELS}")
        
        exp_mode = config.get('exp_mode')
        if exp_mode not in VALID_INTENT_SLOT_EXP_MODE:
            raise ValueError(f"Experiment mode `{exp_mode}` is not recognized! Valid options are: {VALID_INTENT_SLOT_EXP_MODE}")
    else:
        raise NotImplementedError(f"Wrong dataset/task! `{dataset_name}` is not recognized! Valid options are: {VALID_DATASET_NAME}")
    
    lang = config.get('lang', 'all')

    # Refactoring is required... but maybe when we are less busy ;)
    if dataset_name in ["mt560", "nusa"]:
        run_mt(exp_mode, dataset_name, regressor, regressor_json, model, score, lang)
    elif dataset_name in ["intent", "slot"]:
        run_intent_or_slot(exp_mode, dataset_name, regressor, regressor_json, model, score, lang)
