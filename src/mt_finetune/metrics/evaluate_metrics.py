import os
import time
import json
import argparse

# from bleurt import score
import torch

from ..utils.utils import *

COMET_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comet/model/wmt22-comet-da.ckpt")

torch.set_float32_matmul_precision('high')

def _run_all_metrics(config, folder_path):
    logging.debug("=============================")
    config = json.load(open(config))
    lang = config['source_lang'] if config['source_lang'] != "eng" else config['target_lang']
    test_path = os.path.join(DATASET_FOLDER_PATH, "{}/test".format(lang))
    
    # Generate list of paths for the output based on hypothesis
    list_output_path = []
    list_hypothesis_path = []
    list_bleu_output_path = []
    list_sacrebleu_output_path = []
    list_test_tgt_path = []
    list_test_src_path=[]
    list_comet_output_path = []
    for dataset_name in os.listdir(test_path):
        output_path = os.path.join(folder_path, "outputs/output_{}.txt".format(dataset_name))
        hypothesis_path = os.path.join(folder_path, "outputs/output_{}.hyp".format(dataset_name))
        bleu_output_path = os.path.join(folder_path, "outputs/bleu_{}.txt".format(dataset_name))
        sacrebleu_output_path = os.path.join(folder_path, "outputs/sacrebleu_{}.txt".format(dataset_name))
        test_tgt_file_path = os.path.join(test_path, dataset_name, "test_{}.txt".format(config['target_lang']))
        test_src_file_path = os.path.join(test_path, dataset_name, "test_{}.txt".format(config['source_lang']))
        comet_output_path = os.path.join(folder_path, "outputs/comet_{}.json".format(dataset_name))
        
        list_output_path.append(output_path)
        list_hypothesis_path.append(hypothesis_path)
        list_bleu_output_path.append(bleu_output_path)
        list_sacrebleu_output_path.append(sacrebleu_output_path)
        list_test_tgt_path.append(test_tgt_file_path)
        list_test_src_path.append(test_src_file_path)
        list_comet_output_path.append(comet_output_path)

    logging.debug("Start evaluating on multiple metrics")
    for i in range(len(list_output_path)):
        # Grep from hypothesis texts
        c = 'cat {} | grep -P "^H" |sort -V |cut -f 3- > {}'
        command = (c.format(list_output_path[i], list_hypothesis_path[i]))
        execute_command(command)

        # Run evaluation on hypothesis, this will run spBLEU and chrF++
        c = "sacrebleu -tok 'flores200' --metrics bleu chrf --chrf-word-order 2 --confidence -s 'none' {} < {} > {} 2>&1"
        command = (c.format(list_test_tgt_path[i], list_hypothesis_path[i], list_sacrebleu_output_path[i]))
        execute_command(command)
        
        # Run another evaluation on hypothesis, this will run chrF and TER
        c = "sacrebleu -tok 'none' --metrics bleu chrf ter --confidence -s 'none' {} < {} >> {} 2>&1"
        command = (c.format(list_test_tgt_path[i], list_hypothesis_path[i], list_sacrebleu_output_path[i]))
        execute_command(command)
        
        # Run evaluation on comet-score on gpu
        c = "comet-score -s {} -t {} -r {} --model {} --gpus 1 --to_json {}"
        command = (c.format(list_test_src_path[i], list_hypothesis_path[i], list_test_tgt_path[i], COMET_MODEL_PATH, list_comet_output_path[i]))
        execute_command(command)

    logging.debug("Evaluation done")
    logging.debug("=============================")
    
def evaluate_metrics(config_path, stats_path=None):
    folder_path =  os.path.dirname(config_path)
    
    time_start_evaluate = time.time()    

    _run_all_metrics(config_path, folder_path)
    
    time_end_evaluate = time.time()
    
    # Update running time on evaluating metric
    if stats_path is not None:
        stats = json.load(open(stats_path))
        stats['total_evaluate_time'] = time_end_evaluate - time_start_evaluate
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--stats', type=str, required=False, help="Path for the stats run time", default=None)
    args = parser.parse_args()
    
    evaluate_metrics(args.config, args.stats)
