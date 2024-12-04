import argparse

from ..utils.utils import *
from .config import create_config_small100
from .finetune_small100 import run_small100
from ..utils.transform_dataset import transform_dataset
from ..metrics.evaluate_metrics import evaluate_metrics

def main(src_lang, tgt_lang, dataset, size, perform_finetune):
    if perform_finetune:
        exp_dir_name = f"small100_{src_lang}_{tgt_lang}_{dataset}_{size}"
    else:
        exp_dir_name = f"small100_{src_lang}_{tgt_lang}_no_finetune"
    other_lang = src_lang if src_lang != "eng" else tgt_lang

    # Create experiment directories
    create_directories(f"{EXP_DIR}/{exp_dir_name}")

    # Transform dataset
    if perform_finetune:
        transform_dataset(source_language=src_lang, target_language=other_lang, train_dataset=dataset, train_size=size, data_type="txt")

    # Configure
    create_config_small100(src_lang=src_lang, tgt_lang=tgt_lang, dataset=dataset,
                           size=size, perform_finetune=perform_finetune)

    # Finetune if required
    run_small100(config_path=f"{EXP_DIR}/{exp_dir_name}/{exp_dir_name}.json",
                 stats_path=f"{EXP_DIR}/{exp_dir_name}/stats.json")

    # Test
    evaluate_metrics(config_path=f"{EXP_DIR}/{exp_dir_name}/{exp_dir_name}.json",
                     stats_path=f"{EXP_DIR}/{exp_dir_name}/stats.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--src_lang', type=str, required=True, help="Source language")
    parser.add_argument('-tl', '--tgt_lang', type=str, required=True, help="Target language")
    parser.add_argument('-d', '--dataset', type=str, default="", help="Dataset name for training")
    parser.add_argument('-s', '--size', type=int, default=0, help="Size of the dataset")
    parser.add_argument('-ft', '--finetune', type=int, required=True, help="Perform finetuning (0 for no)")
    args = parser.parse_args()
    
    if args.finetune != 0 and (args.dataset == "" or args.size == 0):
        parser.error("-d and -s are required when -ft is not 0")
    
    main(src_lang=args.src_lang, tgt_lang=args.tgt_lang, dataset=args.dataset, size=args.size, perform_finetune=args.finetune)