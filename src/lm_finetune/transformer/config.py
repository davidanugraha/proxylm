import os
import json
import argparse
import logging
from datetime import datetime

from ..utils.utils import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_TRANSFORMER_PATH = os.path.join(CUR_DIR, "config_transformer.json")

def create_config_transformer(src_lang, tgt_lang, dataset, size):
    config = {}
    other_lang = src_lang if src_lang != "eng" else tgt_lang
    config['source_lang'] = src_lang
    config['target_lang'] = tgt_lang
    
    # Folder path refers to the root path for the folder; the rest of the path should be relative path
    folder_name = "_".join(["transformer", src_lang, tgt_lang, dataset, str(size)])
    config['folder_name'] = folder_name
    folder_path = os.path.join(EXP_DIR, folder_name)

    train_path = "{}/train/{}/train_{}_{}.txt"
    dev_path = "{}/dev/{}/dev_{}.txt"

    # sentencepiece
    config['sentencepiece'] = {}
    config['sentencepiece']['input'] = train_path.format(other_lang, dataset, src_lang, str(size)) + ' ' + \
                            train_path.format(other_lang, dataset, tgt_lang, str(size))
    config['sentencepiece']['model_save_path'] = "models/sentencepiece/sentencepiece.bpe"
    config['sentencepiece']['vocab_size'] = str(min(size // 2, 10000))

    # Encoder
    config['encoder'] = {}

    config['encoder']['train'] = {}
    config['encoder']['train']['inputs'] = train_path.format(other_lang, dataset, src_lang, str(size)) + ' ' + \
                            train_path.format(other_lang, dataset, tgt_lang, str(size))
    enc_tr_output_src_path = "encoded/train.bpe.{}".format(src_lang)
    enc_tr_output_trg_path = "encoded/train.bpe.{}".format(tgt_lang)
    config['encoder']['train']['outputs'] = "{} {}".format(enc_tr_output_src_path, enc_tr_output_trg_path)

    config['encoder']['valid'] = {}
    if dataset == "cc_aligned" or dataset.startswith("MT560"):
        config['encoder']['valid']['inputs'] = dev_path.format(other_lang, 'flores', src_lang) + ' ' + \
                                dev_path.format(other_lang, 'flores', tgt_lang)
    else:
        config['encoder']['valid']['inputs'] = dev_path.format(other_lang, dataset, src_lang) + ' ' + \
                                dev_path.format(other_lang, dataset, tgt_lang)
    enc_val_output_src_path = "encoded/valid.bpe.{}".format(src_lang)
    enc_val_output_trg_path = "encoded/valid.bpe.{}".format(tgt_lang)
    config['encoder']['valid']['outputs'] = "{} {}".format(enc_val_output_src_path, enc_val_output_trg_path)
    
    # Preprocess
    config['preprocess'] = {}
    config['preprocess']['trainpref'] = 'encoded/train.bpe'
    config['preprocess']['validpref'] = 'encoded/valid.bpe'
    config['preprocess']['destdir'] = 'preprocess'

    # Training
    with open(CONFIG_TRANSFORMER_PATH, 'r') as f:
        model_param = json.load(f)
    config['train'] = {}
    config['train']['model'] = model_param
    config['train']['tensorboard-logdir'] = 'logs'
    config['train']['save-dir'] = 'models/checkpoints'
    config['train']['best_model_path'] = 'models/checkpoints/checkpoint_best.pt'

    # Saving config file to the experiment folder
    config_json_path = os.path.join(folder_path, '{}.json'.format(folder_name))
    logging.debug("Details of the config file being created")
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
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Dataset name for training")
    parser.add_argument('-s', '--size', type=int, required=True, help="Size of the dataset")
    args = parser.parse_args()

    create_config_transformer(src_lang=args.src_lang, tgt_lang=args.tgt_lang, dataset=args.dataset, size=args.size)
