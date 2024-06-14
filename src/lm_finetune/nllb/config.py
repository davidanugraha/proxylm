import os
import json
import argparse
import logging
from datetime import datetime

import pandas as pd

from ..utils.utils import *

# See README.md
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
NLLB_SPM_MODEL = os.path.join(CUR_DIR, "model", "flores200sacrebleuspm")
PRETRAINED_NLLB_SPM_MODEL = os.path.join(CUR_DIR, "model", "nllb200densedst1bcheckpoint.pt")
FIXED_NLLB_DICTIONARY = os.path.join(CUR_DIR, "model", "nllb200dictionary.txt")
NLLB_LANGUAGE_CODE_MAPPING = os.path.join(CUR_DIR, "model", "language_mapping.csv")
NLLB_DATA_DICTIONARY = os.path.join(CUR_DIR, "model", "nllb_data_dictionary.txt")
NLLB_LANGUAGES = os.path.join(CUR_DIR, "model", "flores200langs.txt")
CONFIG_NLLB_PATH = os.path.join(CUR_DIR, "model", "config_nllb.json")

START_LINE_LANG_DICT = 255998
NLLB_LANGUAGES_NOT_PRETRAINED = ['abs', 'btk', 'bew', 'bhp', 'mad', 'mak', 'mui', 'rej']
        
def _setup_encoder_with_finetune(config, other_lang):
    finetune_path = "{}/train/{}/train_{}_{}.txt"
    dev_path = "{}/dev/{}/dev_{}.txt"

    config['encoder']['finetune'] = {}
    config['encoder']['finetune']['inputs'] = finetune_path.format(other_lang, config['dataset'], config['source_lang'], str(config['size'])) + ' ' + \
                            finetune_path.format(other_lang, config['dataset'], config['target_lang'], str(config['size']))
    enc_tr_output_src_path = "encoded/train.bpe.{}".format(config['source_lang_flores200'])
    enc_tr_output_trg_path = "encoded/train.bpe.{}".format(config['target_lang_flores200'])
    config['encoder']['finetune']['outputs'] = "{} {}".format(enc_tr_output_src_path, enc_tr_output_trg_path)

    config['encoder']['valid'] = {}
    if config['dataset'] == "cc_aligned" or config['dataset'].startswith("MT560"):
        config['encoder']['valid']['inputs'] = dev_path.format(other_lang, 'flores', config['source_lang']) + ' ' + \
                                dev_path.format(other_lang, 'flores', config['target_lang'])
    else:
        config['encoder']['valid']['inputs'] = dev_path.format(other_lang, config['dataset'], config['source_lang']) + ' ' + \
                                dev_path.format(other_lang, config['dataset'], config['target_lang'])
    enc_val_output_src_path = "encoded/valid.bpe.{}".format(config['source_lang_flores200'])
    enc_val_output_trg_path = "encoded/valid.bpe.{}".format(config['target_lang_flores200'])
    config['encoder']['valid']['outputs'] = "{} {}".format(enc_val_output_src_path, enc_val_output_trg_path)
    
def _setup_language_pairs_and_dict(config, folder_path, other_lang):
    # Modify language pairs and write it to models
    with open(FIXED_NLLB_DICTIONARY, 'r') as default_dict_file:
        model_dict_content = default_dict_file.readlines()
    
    # Replace language with dict if it's pretrained, if not add new language as new word
    if config['source_lang'] in NLLB_LANGUAGES_NOT_PRETRAINED:
        index_to_replace = model_dict_content.index("__bjn_Latn__ 1\n")
        model_dict_content[index_to_replace] = f"__{config['source_lang']}_Latn__ 1\n"   
    if config['target_lang'] in NLLB_LANGUAGES_NOT_PRETRAINED:
        index_to_replace = model_dict_content.index("__bug_Latn__ 1\n")
        model_dict_content[index_to_replace] = f"__{config['target_lang']}_Latn__ 1\n"   
    
    # Replace languages
    with open(NLLB_LANGUAGES, 'r') as langs_file:
        config['finetune']['full_languages'] = langs_file.read()
        if config['source_lang'] in NLLB_LANGUAGES_NOT_PRETRAINED:
            config['finetune']['full_languages'] = config['finetune']['full_languages'].replace("bjn_Latn", f"{config['source_lang']}_Latn")
        if config['target_lang'] in NLLB_LANGUAGES_NOT_PRETRAINED:
            config['finetune']['full_languages'] = config['finetune']['full_languages'].replace("bug_Latn", f"{config['target_lang']}_Latn")

    # Open another file for writing
    config['preprocess']['dictionary'] = 'models/nllb200dictionary.txt'
    new_model_dict_path = os.path.join(folder_path, config['preprocess']['dictionary'])
    logging.debug("new model dict file path: %s", new_model_dict_path)
    with open(new_model_dict_path, 'w') as new_model_dict_file:
        new_model_dict_file.writelines(model_dict_content)
    
def _setup_config_finetune_model(config):
    with open(CONFIG_NLLB_PATH, 'r') as f:
        model_param = json.load(f)
    config['finetune']['model'] = model_param
    config['finetune']['tensorboard-logdir'] = 'logs'
    config['finetune']['save-dir'] = 'models/checkpoints'
    config['finetune']['best_model_path'] = 'models/checkpoints/checkpoint_best.pt'

def create_config_nllb(src_lang, tgt_lang, dataset, size, perform_finetune):
    config = {}
    other_lang = src_lang if src_lang != "eng" else tgt_lang

    # Convert source_lang and target_lang to appropriate NLLB-Lang code
    config['source_lang'] = src_lang
    config['target_lang'] = tgt_lang
    lang_map_df = pd.read_csv(NLLB_LANGUAGE_CODE_MAPPING)
    config['source_lang_flores200'] = lang_map_df[lang_map_df['trimmed_code'] == config['source_lang']]['flores200_code'].unique()[0]
    config['target_lang_flores200'] = lang_map_df[lang_map_df['trimmed_code'] == config['target_lang']]['flores200_code'].unique()[0]
    config['dataset'] = dataset
    config['size'] = size

    # Folder path refers to the root path for the folder; the rest of the path should be relative path
    if perform_finetune:
        folder_name = "_".join(["nllb", src_lang, tgt_lang, dataset, str(size)])
    else:
        folder_name = f"nllb_{src_lang}_{tgt_lang}_no_finetune"
    config['folder_name'] = folder_name
    folder_path = os.path.join(EXP_DIR, folder_name)
    
    # Sentencepiece model
    config['sentencepiece'] = {}
    config['sentencepiece']['model_save_path'] = NLLB_SPM_MODEL
    
    # Encoder
    config['encoder'] = {}
    
    # Preprocess
    config['preprocess'] = {}
    config['preprocess']['destdir'] = 'preprocess'

    # Finetune
    config['finetune'] = {}
    config['finetune']['pretrained_model_path'] = PRETRAINED_NLLB_SPM_MODEL

    if perform_finetune != 0:
        config['perform_finetune'] = True
        
        # Additional setup for encoder and preprocess
        _setup_encoder_with_finetune(config, other_lang)
        config['preprocess']['trainpref'] = 'encoded/train.bpe'
        config['preprocess']['validpref'] = 'encoded/valid.bpe'
        _setup_config_finetune_model(config)
    else:
        config['perform_finetune'] = False
    
    _setup_language_pairs_and_dict(config, folder_path)
            
    # Saving config file to the experiment folder
    config_json_path = os.path.join(folder_path, '{}.json'.format(folder_name))
    logging.debug(f"Details of the config file being created at {config_json_path}")
    logging.debug(config)
    with open(config_json_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Provide statistics on runtime
    stats = {}
    stats['time_created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats_json_path = os.path.join(folder_path, 'stats.json')
    with open(stats_json_path, 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--src_lang', type=str, required=True, help="Source language")
    parser.add_argument('-tl', '--tgt_lang', type=str, required=True, help="Target language")
    parser.add_argument('-d', '--dataset', type=str, default="", help="Dataset name for training")
    parser.add_argument('-s', '--size', type=int, default=0, help="Size of the dataset")
    parser.add_argument('-ft', '--finetune', type=int, required=True, help="Perform finetuning (0 for no)")
    args = parser.parse_args()
    
    if args.finetune != 0 and (args.dataset == "" or args.size == 0):
        parser.error("-d and -s are required when -ft is not 0")

    create_config_nllb(src_lang=args.src_lang, tgt_lang=args.tgt_lang,
                           dataset=args.dataset, size=args.size, perform_finetune=args.finetune)
