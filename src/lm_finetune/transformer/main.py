import argparse

from ..utils.utils import *
from .config import create_config_transformer
from .train_transformer import train_transformer
from ..utils.transform_dataset import transform_dataset
from ..metrics.evaluate_metrics import evaluate_metrics

def main(src_lang, tgt_lang, dataset, size):
    exp_dir_name = f"transformer_{src_lang}_{tgt_lang}_{dataset}_{size}"
    other_lang = src_lang if src_lang != "eng" else tgt_lang

    # Create experiment directories
    create_directories(f"{EXP_DIR}/{exp_dir_name}")

    # Transform dataset
    transform_dataset(source_language=src_lang, target_language=other_lang, train_dataset=dataset, train_size=size, data_type="txt")

    # Configure
    create_config_transformer(src_lang=src_lang, tgt_lang=tgt_lang, dataset=dataset, size=size)

    # Train
    train_transformer(config_path=f"{EXP_DIR}/{exp_dir_name}/{exp_dir_name}.json",
                      stats_path=f"{EXP_DIR}/{exp_dir_name}/stats.json")

    # Test
    evaluate_metrics(config_path=f"{EXP_DIR}/{exp_dir_name}/{exp_dir_name}.json",
                     stats_path=f"{EXP_DIR}/{exp_dir_name}/stats.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--src_lang', type=str, required=True, help="Source language")
    parser.add_argument('-tl', '--tgt_lang', type=str, required=True, help="Target language")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Dataset name for training")
    parser.add_argument('-s', '--size', type=int, required=True, help="Size of the dataset")
    args = parser.parse_args()

    main(src_lang=args.src_lang, tgt_lang=args.tgt_lang, dataset=args.dataset, size=args.size)