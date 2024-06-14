import argparse
import csv
import os
import json
import re
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

def get_test_outputs(experiment_dir):
    '''modified to take in additional dir layer for language'''
    entries = []
    tmp_entries_dict = {}
    logging.debug(f"start processing lang folders under {experiment_dir}")
    for lang_dir in os.listdir(experiment_dir):
        lang_dir_path=os.path.join(experiment_dir, lang_dir)
        logging.debug(f"- processing {lang_dir_path}")
        for exp_subdir in os.listdir(lang_dir_path):
            exp_subdir_path = os.path.join(lang_dir_path, exp_subdir)
            output_dir = os.path.join(exp_subdir_path, "outputs")
            if os.path.exists(os.path.join(exp_subdir_path, "{}.json".format(exp_subdir))) and \
                os.path.exists(output_dir):
                files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
                for file in files:
                    if file.startswith("sacrebleu_") and file.endswith(".txt"):
                        file_path = os.path.join(output_dir, file)
                        entry = parse_txt_file(file_path)
                        entry_key=file.split('_', 1)[1].replace('.txt','')
                        tmp_entries_dict[entry_key]=entry
                
                # parse a second round and add comet score stats to entry
                for file in files:
                    if file.startswith('comet_') and file.endswith('.json'):
                        entry_key=file.split('_', 1)[1].replace('.json','')
                        if entry_key in tmp_entries_dict.keys():
                            comet_file_path = os.path.join(output_dir, file)
                            entries.append(add_comet_score_stats_to_entry(tmp_entries_dict[entry_key],comet_file_path))
    logging.debug(f"processed {len(os.listdir(experiment_dir))} lang folders")
    return entries

def parse_txt_file(file_path):
    with open(file_path, "r") as file:
        source_lang, target_lang, dataset_name, dataset_size = parse_folder_name(file_path)
        test_dataset = os.path.basename(file_path).replace("sacrebleu_", "").replace(".txt", "")
        row_entry = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "train_dataset": dataset_name,
            "dataset_size": dataset_size,
            "test_dataset": test_dataset,
        }
        content = file.read()

        # There are 2 json objects; doing json.loads immediately seems breaking the loader
        json_objects = content.split('[\n')
        
        for json_obj in json_objects:
            # Check if it starts as valid json dictionary
            if len(json_obj) == 0 or json_obj[0] != "{":
                continue
            parse_json_obj = '[\n'+ json_obj
            parse_json_obj = parse_json_obj.split(']\n')[0]
            if len(parse_json_obj) == 0:
                continue
            # Recover the lost ']' due to split
            metric_list = json.loads(parse_json_obj + ']')
            for metric in metric_list:
                # Special rule, we name metric spBLEU if name is BLEU and has 'tok' parameter with 'spm-flores'
                metric_name = "spBLEU" if (metric['name'] == 'BLEU' and metric['tok'] == 'flores200') else metric['name']
                
                # Check if it exists and insert to the row
                row_entry[f"{metric_name}_mean"] = metric["confidence_mean"] if "confidence_mean" in metric else 0
                row_entry[f"{metric_name}_se"] = metric["confidence_var"] if "confidence_var" in metric else 0

        return row_entry

def add_comet_score_stats_to_entry(entry, comet_json_path):
    comet_scores=[]
    with open(comet_json_path,'r',encoding="utf8") as f:
        lines=f.readlines()
        for line in lines:
            pattern = r'"COMET": (\d+\.\d+)'
            match = re.search(pattern, line)
            if match:
                comet_scores.append(float(match.group(1)))
    if len(comet_scores)!=0:
        comet_scores = np.array(comet_scores)
        entry['comet_score_mean']=round(np.mean(comet_scores), 4)
        entry['comet_score_variance']=round(np.sqrt(np.var(comet_scores)), 4) 
    return entry

def parse_folder_name(file_path):
    folder_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    parts = folder_name.split("_")
    return parts[0], parts[1], "_".join(parts[2:-1]), parts[-1]

def write_to_csv(entries, output_file):
    logging.debug(f"saved total {len(entries)} entries to {output_file}")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=entries[0].keys())
        writer.writeheader()
        writer.writerows(entries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True, help="path to experiment dir")
    parser.add_argument('--csv_output', type=str, required=True, help="Path to CSV file to save")
    args = parser.parse_args()

    entries = get_test_outputs(args.experiment_dir)
    write_to_csv(entries, args.csv_output)