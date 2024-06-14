import os
import json
import argparse
import logging
from datetime import datetime

import pandas as pd

from ..utils.utils import *

# See README.md
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
M2M100_SPM_MODEL = os.path.join(CUR_DIR, "model", "spm.128k")
PRETRAINED_M2M100_SPM_MODEL = os.path.join(CUR_DIR, "model", "1.2B_last_checkpoint.pt")
DEFAULT_DATA_DICT = os.path.join(CUR_DIR, "model", "data_dict.128k.txt")
M2M100_DEFAULT_DICT = os.path.join(CUR_DIR, "model", "model_dict.128k.txt")
M2M100_LANGUAGE_PAIR_MODEL = os.path.join(CUR_DIR, "model", "language_pairs.txt")
M2M100_LANGUAGE_CODE_MAPPING = os.path.join(CUR_DIR, "model", "language_mapping.csv")

CONFIG_M2M100_PATH = os.path.join(CUR_DIR, "model", "config_m2m100.json")
START_LINE_LANG_DICT = 128000
MADEUP_WORDS_LIST = ["madeupwordforbt 1", "madeupword0000 0", "madeupword0001 0", "madeupword0002 0",
                     "madeupword0003 0", "madeupword0004 0", "madeupword0005 0", "madeupword0006 0"]

LANGUAGES_NOT_PRETRAINED = ['asm', 'dik', 'ewe', 'fao', 'hne', 'kab', 'kin', 'kir', 'lmo', 'mri',
                           'sna', 'tat', 'tel', 'tgk', 'tuk', 'uig', 'abs', 'btk', 'bew', 'bhp',
                           'mad', 'mak', 'min', 'muj', 'rej']

def _setup_language_pairs_and_dict(config, folder_path):
    # Modify language pairs and write it to models
    with open(M2M100_LANGUAGE_PAIR_MODEL, 'r') as lang_pair_file, \
         open(M2M100_DEFAULT_DICT, 'r') as default_dict_file:
        pair_content = lang_pair_file.read()
        model_dict_content = default_dict_file.readlines()

    lang_map_df = pd.read_csv(M2M100_LANGUAGE_CODE_MAPPING)
    m2m100_src_lang = lang_map_df[lang_map_df['flores_abbr'] == config['source_lang']]['m2m100_abbr'].unique()[0]
    m2m100_tgt_lang = lang_map_df[lang_map_df['flores_abbr'] == config['target_lang']]['m2m100_abbr'].unique()[0]

    # Modify the content based on the flores language abbreviation
    if not pd.isna(m2m100_src_lang):
        pair_content = pair_content.replace(m2m100_src_lang, config['source_lang'])
    if not pd.isna(m2m100_tgt_lang):
        pair_content = pair_content.replace(m2m100_tgt_lang, config['target_lang'])
    
    # If pair is not found, add to pair_content
    if f"{config['source_lang']}-{config['target_lang']}" not in pair_content:
        pair_content = pair_content[:-1] + f",{config['source_lang']}-{config['target_lang']}\n"
    if f"{config['target_lang']}-{config['source_lang']}" not in pair_content:
        pair_content = pair_content[:-1] + f",{config['target_lang']}-{config['source_lang']}\n"
    
    # Write to new language pair file path
    config['finetune']['language_pair'] = 'models/language_pairs.txt'
    new_path_lang_pair_model_path = os.path.join(folder_path, config['finetune']['language_pair'])
    with open(new_path_lang_pair_model_path, 'w') as new_lang_pair_file:
        new_lang_pair_file.write(pair_content)
    
    # Replace language with dict if it's pretrained, if not add new language as new word
    if config['source_lang'] in LANGUAGES_NOT_PRETRAINED:
        model_dict_content.insert(START_LINE_LANG_DICT + 100, f"__{config['source_lang']}__ 1\n")
        model_dict_content.pop()
    else:
        for i in range(START_LINE_LANG_DICT, len(model_dict_content)):
            if model_dict_content[i] == f"__{m2m100_src_lang}__ 1\n":    
                model_dict_content[i] = f"__{config['source_lang']}__ 1\n"
                break
            
    if config['target_lang'] in LANGUAGES_NOT_PRETRAINED:
        model_dict_content.insert(START_LINE_LANG_DICT + 100, f"__{config['target_lang']}__ 1\n")
        model_dict_content.pop()
    else:
        for i in range(START_LINE_LANG_DICT, len(model_dict_content)):
            if model_dict_content[i] == f"__{m2m100_tgt_lang}__ 1\n":    
                model_dict_content[i] = f"__{config['target_lang']}__ 1\n"
                break

    # Open another file for writing
    config['finetune']['dictionary'] = 'models/model_dict.txt'
    new_model_dict_path = os.path.join(folder_path, config['finetune']['dictionary'])

    with open(new_model_dict_path, 'w') as new_model_dict_file:
        new_model_dict_file.writelines(model_dict_content)
        
def _setup_encoder_with_finetune(config, other_lang):
    finetune_path = "{}/train/{}/train_{}_{}.txt"
    dev_path = "{}/dev/{}/dev_{}.txt"

    config['encoder']['finetune'] = {}
    config['encoder']['finetune']['inputs'] = finetune_path.format(other_lang, config['dataset'], config['source_lang'], str(config['size'])) + ' ' + \
                            finetune_path.format(other_lang, config['dataset'], config['target_lang'], str(config['size']))
    enc_tr_output_src_path = "encoded/train.bpe.{}".format(config['source_lang'])
    enc_tr_output_trg_path = "encoded/train.bpe.{}".format(config['target_lang'])
    config['encoder']['finetune']['outputs'] = "{} {}".format(enc_tr_output_src_path, enc_tr_output_trg_path)

    config['encoder']['valid'] = {}
    if config['dataset'] == "cc_aligned" or config['dataset'].startswith("MT560"):
        config['encoder']['valid']['inputs'] = dev_path.format(other_lang, 'flores', config['source_lang']) + ' ' + \
                                dev_path.format(other_lang, 'flores', config['target_lang'])
    else:
        config['encoder']['valid']['inputs'] = dev_path.format(other_lang, config['dataset'], config['source_lang']) + ' ' + \
                                dev_path.format(other_lang, config['dataset'], config['target_lang'])
    enc_val_output_src_path = "encoded/valid.bpe.{}".format(config['source_lang'])
    enc_val_output_trg_path = "encoded/valid.bpe.{}".format(config['target_lang'])
    config['encoder']['valid']['outputs'] = "{} {}".format(enc_val_output_src_path, enc_val_output_trg_path)
    
def _setup_config_finetune_model(config):
    with open(CONFIG_M2M100_PATH, 'r') as f:
        model_param = json.load(f)
    config['finetune']['model'] = model_param
    config['finetune']['tensorboard-logdir'] = 'logs'
    config['finetune']['save-dir'] = 'models/checkpoints'
    config['finetune']['best_model_path'] = 'models/checkpoints/checkpoint_best.pt'

def create_config_m2m100(src_lang, tgt_lang, dataset, size, perform_finetune):
    config = {}
    other_lang = src_lang if src_lang != "eng" else tgt_lang
    config['source_lang'] = src_lang
    config['target_lang'] = tgt_lang
    config['dataset'] = dataset
    config['size'] = size

    # Folder path refers to the root path for the folder; the rest of the path should be relative path
    if perform_finetune:
        folder_name = "_".join(["m2m100", src_lang, tgt_lang, dataset, str(size)])
    else:
        folder_name = f"m2m100_{src_lang}_{tgt_lang}_no_finetune"
    config['folder_name'] = folder_name
    folder_path = os.path.join(EXP_DIR, folder_name)
    
    # Sentencepiece model
    config['sentencepiece'] = {}
    config['sentencepiece']['model_save_path'] = M2M100_SPM_MODEL
    
    # Encoder
    config['encoder'] = {}
    
    # Preprocess
    config['preprocess'] = {}
    config['preprocess']['destdir'] = 'preprocess'
    config['preprocess']['dictionary'] = 'models/data_dict.txt'
    new_data_dict_path = os.path.join(folder_path, config['preprocess']['dictionary'])
    # Copy preprocessing dictionary
    with open(DEFAULT_DATA_DICT, 'r') as data_dict_file, open(new_data_dict_path, 'w') as new_data_dict_file:
        new_data_dict_file.writelines(data_dict_file.readlines())
        if perform_finetune:
            # If perform finetune, we use data dict and add madeup words to fix embedding issue alignment
            if src_lang in LANGUAGES_NOT_PRETRAINED and tgt_lang in LANGUAGES_NOT_PRETRAINED:
                # Assuming english centric, need to add this to the vocab
                selected_words_list = MADEUP_WORDS_LIST[:-2]
            elif src_lang in LANGUAGES_NOT_PRETRAINED or tgt_lang in LANGUAGES_NOT_PRETRAINED:
                selected_words_list = MADEUP_WORDS_LIST[:-1]
            else:
                selected_words_list = MADEUP_WORDS_LIST
            for madeup_words in selected_words_list:
                new_data_dict_file.write(madeup_words + "\n")

    # Finetune
    config['finetune'] = {}
    config['finetune']['pretrained_model_path'] = PRETRAINED_M2M100_SPM_MODEL
    
    if perform_finetune != 0:
        config['perform_finetune'] = True
    else:
        config['perform_finetune'] = False

    _setup_language_pairs_and_dict(config, folder_path)

    if perform_finetune != 0:
        config['perform_finetune'] = True
        
        # Additional setup for encoder and preprocess
        _setup_encoder_with_finetune(config, other_lang)
        config['preprocess']['trainpref'] = 'encoded/train.bpe'
        config['preprocess']['validpref'] = 'encoded/valid.bpe'
        _setup_config_finetune_model(config)
    else:
        config['perform_finetune'] = False
            
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

    create_config_m2m100(src_lang=args.src_lang, tgt_lang=args.tgt_lang,
                           dataset=args.dataset, size=args.size, perform_finetune=args.finetune)