import os
import json
import argparse
import time
import re

from ..utils.utils import *

def _encode(config, folder_path, mode='finetune'):
    c = "python {} --model {}.model --output_format=piece --inputs {} --outputs {} --min-len 0"
    spm_save_abs_path = os.path.join(folder_path, config['sentencepiece']['model_save_path'])

    if mode != "test" and len(config["encoder"]) != 0:
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
    elif mode == "test":
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

    if config['perform_finetune'] == True:
        # Need to preprocess finetune and valid 
        c = "fairseq-preprocess --source-lang {} --target-lang {} \
            --trainpref {} --validpref {} --testpref {} --destdir {} \
            --thresholdsrc 0 --thresholdtgt 0 \
            --srcdict {} --tgtdict {}"
        command = (c.format(config['source_lang'], config['target_lang'],
                os.path.join(folder_path, config['preprocess']['trainpref']),
                os.path.join(folder_path, config['preprocess']['validpref']),
                ",".join(test_pref),
                os.path.join(folder_path, config['preprocess']['destdir']),
                os.path.join(folder_path, config['preprocess']['dictionary']),
                os.path.join(folder_path, config['preprocess']['dictionary'])))
    else:
        # No need to preprocess finetune and valid 
        c = "fairseq-preprocess --source-lang {} --target-lang {} \
            --testpref {} --destdir {} \
            --thresholdsrc 0 --thresholdtgt 0 \
            --srcdict {} --tgtdict {}"
        command = (c.format(config['source_lang'], config['target_lang'],
                ",".join(test_pref),
                os.path.join(folder_path, config['preprocess']['destdir']),
                os.path.join(folder_path, config['preprocess']['dictionary']),
                os.path.join(folder_path, config['preprocess']['dictionary'])))

    execute_command(command)

def _finetune(config, folder_path):
    # Get languages used for finetune based on language pair text
    with open(os.path.join(folder_path, config['finetune']['language_pair']), 'r') as lang_pair_file:
        language_pairs = lang_pair_file.read()
    all_languages_codes = set(re.findall(r'\b[a-z]{2,3}\b', language_pairs))
    languages_used = ",".join(all_languages_codes)

    # Config taken from https://medium.com/@juanluis1702/how-to-fine-tune-m2m-100-model-in-fairseq-6670676ddf2b
    # transformer_wmt_en_de_big is used, check https://github.com/facebookresearch/fairseq/issues/3233
    command = f"fairseq-train {os.path.join(folder_path, config['preprocess']['destdir'])}\
        --source-lang {config['source_lang']} \
        --target-lang {config['target_lang']} \
        --arch transformer_wmt_en_de_big \
        --share-decoder-input-output-embed --share-all-embeddings \
        --encoder-layers {config['finetune']['model']['encoder-layers']} \
        --decoder-layers {config['finetune']['model']['decoder-layers']} \
        --encoder-embed-dim {config['finetune']['model']['encoder-embed-dim']} \
        --decoder-embed-dim {config['finetune']['model']['decoder-embed-dim']} \
        --encoder-ffn-embed-dim {config['finetune']['model']['encoder-ffn-embed-dim']} \
        --decoder-ffn-embed-dim {config['finetune']['model']['decoder-ffn-embed-dim']} \
        --encoder-attention-heads {config['finetune']['model']['encoder-attention-heads']} \
        --decoder-attention-heads {config['finetune']['model']['decoder-attention-heads']} \
        --encoder-normalize-before --decoder-normalize-before \
        --encoder-layerdrop 0.05 --decoder-layerdrop 0.05 \
        --task translation_multi_simple_epoch \
        --langs {languages_used} \
        --lang-pairs {config['source_lang']}-{config['target_lang']} \
        --encoder-langtok tgt \
        --dropout {config['finetune']['model']['dropout']} \
        --attention-dropout {config['finetune']['model']['attention-dropout']} \
        --relu-dropout {config['finetune']['model']['relu-dropout']} \
        --weight-decay {config['finetune']['model']['weight-decay']} \
        --label-smoothing {config['finetune']['model']['label-smoothing']} --criterion label_smoothed_cross_entropy \
        --optimizer adam --adam-eps 1e-6 --adam-betas '(0.9, 0.98)' --clip-norm {config['finetune']['model']['clip-norm']} \
        --lr-scheduler inverse_sqrt --warmup-updates {config['finetune']['model']['warmup-updates']} \
        --lr {config['finetune']['model']['lr']} --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
        --batch-size {config['finetune']['model']['batch-size']} \
        --update-freq {config['finetune']['model']['update-freq']} \
        --max-epoch {config['finetune']['model']['max-epoch']} --no-epoch-checkpoints \
        --tensorboard-logdir {os.path.join(folder_path, config['finetune']['tensorboard-logdir'])} \
        --save-dir {os.path.join(folder_path, config['finetune']['save-dir'])} \
        --finetune-from-model {config['finetune']['pretrained_model_path']} \
        --max-tokens {config['finetune']['model']['max-tokens']} --memory-efficient-fp16 --ddp-backend no_c10d --patience 6"
    execute_command(command)

def _generate_hypothesis(config, folder_path):
    lang = config['source_lang'] if config['source_lang'] != "eng" else config['target_lang']
    test_path = os.path.join(DATASET_FOLDER_PATH, "{}/test".format(lang))
    
    # Generate list of paths for the output based on hypothesis
    list_output_path = []
    for dataset_name in os.listdir(test_path):
        output_path = os.path.join(folder_path, "outputs/output_{}.txt".format(dataset_name))
        list_output_path.append(output_path)
    
    if config["perform_finetune"] == True:
        model_path = os.path.join(folder_path, config['finetune']['best_model_path'])
    else:
        model_path = config['finetune']['pretrained_model_path']

    for i in range(len(list_output_path)):
        # Generate output from test that will be generated to hypothesis
        c = 'fairseq-generate \
            {} \
            --batch-size {} \
            --source-lang {} --target-lang {} \
            --path {} \
            --fixed-dictionary {} \
            --beam 5 --lenpen 1.2 \
            --task translation_multi_simple_epoch \
            --encoder-langtok tgt \
            --lang-pairs {}-{} \
            --gen-subset test{} \
            --remove-bpe=sentencepiece > {} 2>&1'
        command = (c.format(
            os.path.join(folder_path, config['preprocess']['destdir']),
            config['finetune']['model']['batch-size'],
            config['source_lang'], config['target_lang'],
            model_path,
            os.path.join(folder_path, config['finetune']['dictionary']),
            config['source_lang'], config['target_lang'],
            "" if i == 0 else str(i),
            list_output_path[i]
        ))
        
        execute_command(command)

# Run SMaLL100 while checking whether fine-tuned or 
def run_small100(config_path, stats_path):
    logging.debug("=============================")
    time_start_finetune = time.time()

    folder_path = os.path.dirname(config_path)
    config = json.load(open(config_path))

    try:
        _encode(config, folder_path, 'finetune')
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

    if config['perform_finetune'] == True:
        try:
            _finetune(config, folder_path)
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
    
    time_end_finetune = time.time()
    
    if stats_path is not None:
        stats = json.load(open(stats_path))
        stats['total_finetune_time'] = time_end_finetune - time_start_finetune
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)

    logging.debug("=============================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to config file")
    parser.add_argument('-s', '--stats', type=str, required=False, help="Path for the stats run time", default=None)
    args = parser.parse_args()

    run_small100(config_path=args.config, stats_path=args.stats)