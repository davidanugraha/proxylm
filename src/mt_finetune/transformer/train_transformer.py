import os
import json
import argparse
import time

from ..utils.utils import *

def _create_sentencepiece(config, folder_path):
    c = "python {}  \
        --input={} \
        --model_prefix={} \
        --vocab_size={} \
        --character_coverage=1.0 \
        --model_type=bpe"
        
    # Split input and join with DATASET_FOLDER_PATH
    spm_input = config['sentencepiece']['input'].split(" ")
    spm_input_with_path = [os.path.join(DATASET_FOLDER_PATH, file_name) for file_name in spm_input]
    spm_input_joined = ",".join(spm_input_with_path)

    command = (c.format(
        SPM_SCRIPT_PATH,
        spm_input_joined,
        os.path.join(folder_path, config['sentencepiece']['model_save_path']),
        config['sentencepiece']['vocab_size']
    ))
    execute_command(command)

def _encode(config, folder_path, mode='train'):
    c = "python {} --model {} --output_format=piece --inputs {} --outputs {} --min-len 0"
    spm_save_abs_path = os.path.join(folder_path, f"{config['sentencepiece']['model_save_path']}.model")

    if mode != "test":
        # Split encoder inputs and join with folderpath
        encoder_inputs = config['encoder'][mode]['inputs'].split(" ")
        encoder_inputs_with_path = [os.path.join(DATASET_FOLDER_PATH, file_name) for file_name in encoder_inputs]
        encoder_inputs_joined = " ".join(encoder_inputs_with_path)
        
        # Split encoder outputs and join with folderpath
        encoder_outputs = config['encoder'][mode]['outputs'].split(" ")
        encoder_outputs_with_path = [os.path.join(folder_path, file_name) for file_name in encoder_outputs]
        encoder_outputs_joined = " ".join(encoder_outputs_with_path)

        command = (c.format(ENCODER_SCRIPT_PATH, spm_save_abs_path,
                    encoder_inputs_joined, encoder_outputs_joined))
        execute_command(command)
    else:
        # For each test folder, we encode
        lang = config['source_lang'] if config['source_lang'] != "eng" else config['target_lang']
        test_path = os.path.join(DATASET_FOLDER_PATH, "{}/test".format(lang))
        for dataset_name in os.listdir(test_path):
            encoder_input = os.path.join(test_path, dataset_name, "test_{}.txt".format(config['source_lang'])) + ' ' + \
                os.path.join(test_path, dataset_name, "test_{}.txt".format(config['target_lang']))
            encoder_output = os.path.join(folder_path, "encoded/test_{}.bpe.{}".format(dataset_name, config['source_lang'])) + ' ' + \
                os.path.join(folder_path, "encoded/test_{}.bpe.{}".format(dataset_name, config['target_lang']))
            command = (c.format(ENCODER_SCRIPT_PATH, spm_save_abs_path, encoder_input, encoder_output))
            execute_command(command)

def _preprocess(config, folder_path):
    lang = config['source_lang'] if config['source_lang'] != "eng" else config['target_lang']
    test_path = os.path.join(DATASET_FOLDER_PATH, "{}/test".format(lang))
    test_pref = []
    for dataset_name in os.listdir(test_path):
        test_pref.append(os.path.join(folder_path, "encoded/test_{}.bpe".format(dataset_name)))

    c = "fairseq-preprocess --source-lang {} --target-lang {} \
        --trainpref {} --validpref {} --testpref {} --destdir {} \
        --joined-dictionary"
    command = (c.format(config['source_lang'], config['target_lang'],
            os.path.join(folder_path, config['preprocess']['trainpref']),
            os.path.join(folder_path, config['preprocess']['validpref']),
            ",".join(test_pref),
            os.path.join(folder_path, config['preprocess']['destdir'])))
    execute_command(command)

def _train(config, folder_path):
    c = "fairseq-train \
        {}\
        --source-lang {} --target-lang {} \
        --arch transformer --share-all-embeddings \
        --encoder-layers {} --decoder-layers {} \
        --encoder-embed-dim {} --decoder-embed-dim {} \
        --encoder-ffn-embed-dim {} --decoder-ffn-embed-dim {} \
        --encoder-attention-heads {} --decoder-attention-heads {} \
        --encoder-normalize-before --decoder-normalize-before \
        --dropout {} --attention-dropout {} --relu-dropout {} \
        --weight-decay {} \
        --label-smoothing {} --criterion label_smoothed_cross_entropy \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm {} \
        --lr-scheduler inverse_sqrt --warmup-updates {} --warmup-init-lr 1e-7 \
        --lr {} --stop-min-lr 1e-9 \
        --batch-size {} \
        --update-freq {} \
        --max-epoch {} \
        --tensorboard-logdir {} \
        --save-dir {} \
        --no-epoch-checkpoints \
        --patience 6"
    command = (c.format(
            os.path.join(folder_path, config['preprocess']['destdir']),
            config['source_lang'], config['target_lang'],
            config['train']['model']['encoder-layers'], config['train']['model']['decoder-layers'],
            config['train']['model']['encoder-embed-dim'], config['train']['model']['decoder-embed-dim'],
            config['train']['model']['encoder-ffn-embed-dim'], config['train']['model']['decoder-ffn-embed-dim'],
            config['train']['model']['encoder-attention-heads'], config['train']['model']['decoder-attention-heads'],
            config['train']['model']['dropout'], config['train']['model']['attention-dropout'],
            config['train']['model']['relu-dropout'], config['train']['model']['weight-decay'],
            config['train']['model']['label-smoothing'], config['train']['model']['clip-norm'],
            config['train']['model']['warmup-updates'], config['train']['model']['lr'],
            config['train']['model']['batch-size'], config['train']['model']['update-freq'], config['train']['model']['max-epoch'],
            os.path.join(folder_path, config['train']['tensorboard-logdir']),
            os.path.join(folder_path, config['train']['save-dir'])
    ))

    execute_command(command)

def _generate_hypothesis(config, folder_path):
    lang = config['source_lang'] if config['source_lang'] != "eng" else config['target_lang']
    test_path = os.path.join(DATASET_FOLDER_PATH, "{}/test".format(lang))
    
    # Generate list of paths for the output based on hypothesis
    list_output_path = []
    for dataset_name in os.listdir(test_path):
        output_path = os.path.join(folder_path, "outputs/output_{}.txt".format(dataset_name))
        list_output_path.append(output_path)

    for i in range(len(list_output_path)):
        # Generate output from test that will be generated to hypothesis
        c = 'fairseq-generate \
            {} \
            --source-lang {} --target-lang {} \
            --path {} \
            --beam 5 --lenpen 1.2 \
            --gen-subset test{} \
            --remove-bpe=sentencepiece > {} 2>&1'
        command = (c.format(
            os.path.join(folder_path, config['preprocess']['destdir']),
            config['source_lang'], config['target_lang'],
            os.path.join(folder_path, config['train']['best_model_path']),
            "" if i == 0 else str(i),
            list_output_path[i]
        ))
        execute_command(command)

def train_transformer(config_path, stats_path):
    logging.debug("=============================")
    time_start_train = time.time()

    folder_path = os.path.dirname(config_path)
    config = json.load(open(config_path))
    try:
        _create_sentencepiece(config, folder_path)
        logging.debug("Sentence piece model creation successful!")
    except Exception as e:
        logging.error("Sentence piece creation failed with error %s", str(e), exc_info=True)
        raise

    try:
        _encode(config, folder_path, 'train')
        _encode(config, folder_path, 'valid')
        _encode(config, folder_path, 'test')
        logging.debug("Encoding dataset creation successful!")
    except Exception as e:
        logging.error("Encoding dataset failed with error: %s", str(e), exc_info=True)
        raise

    try:
        _preprocess(config, folder_path)
        logging.debug("Preprocessing successful!")
    except Exception as e:
        logging.error("Preprocessing dataset failed with error: %s", str(e), exc_info=True)
        raise

    try:
        _train(config, folder_path)
        logging.debug("Training successful!")
    except Exception as e:
        logging.error("Training failed with error: %s", str(e), exc_info=True)
        raise
    
    try:
        _generate_hypothesis(config, folder_path)
        logging.debug("Generating hypothesis successful!")
    except Exception as e:
        logging.error("Generating hypothesis failed with error: %s", str(e), exc_info=True)
        raise
    
    time_end_train = time.time()
    
    if stats_path is not None:
        stats = json.load(open(stats_path))
        stats['total_train_time'] = time_end_train - time_start_train
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)

    logging.debug("=============================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to config file")
    parser.add_argument('-s', '--stats', type=str, required=False, help="Path for the stats run time", default=None)
    args = parser.parse_args()

    train_transformer(config_path=args.config, stats_path=args.stats)