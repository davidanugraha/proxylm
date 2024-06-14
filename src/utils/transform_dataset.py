import os
import json
import argparse
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
DATASET_FOLDER_NAME = "dataset"
DATASET_FOLDER_PATH = os.path.join(EXP_DIR, DATASET_FOLDER_NAME)

logging.basicConfig(level=logging.DEBUG)

CATEGORIES = ["train", "test", "dev"]

def _convert_dataset_from_json(language, train_dataset, train_size):
    for category in CATEGORIES:
        if category == "train":
            source_file_name = "{}_{}_{}.txt".format(category, language, train_size)
            target_file_name = "{}_eng_{}.txt".format(category, train_size)
        else:
            source_file_name = "{}_{}.txt".format(category, language)
            target_file_name = "{}_eng.txt".format(category)

        directory = os.path.join(DATASET_FOLDER_PATH, "{}/{}".format(language, category))

        for named_dataset in os.listdir(directory):
            dataset_path = os.path.join(directory, named_dataset)

            if category != "train" or os.path.basename(named_dataset) == train_dataset:
                source_file_path = os.path.join(dataset_path, source_file_name)
                target_file_path = os.path.join(dataset_path, target_file_name)

                # Cannot find .json file in the folder
                json_translation_path = os.path.join(dataset_path, "{}.json".format(category))
                if not os.path.isfile(json_translation_path):
                    logging.error(f"Please put {category}.json in directory {dataset_path}.")

                if os.path.exists(source_file_path) or os.path.exists(target_file_path):
                    logging.debug(f"{target_file_name} or {source_file_name} already exists, skipping.")
                else:
                    logging.debug(f"Generating {target_file_name}, {source_file_name}...")
                    
                    with open(json_translation_path, 'r') as json_file:
                        data = json_file.readlines()

                    with open(source_file_path, 'w', encoding='utf-8') as source_file, open(target_file_path, 'w', encoding='utf-8') as target_file:
                        read = 0
                        for line in data:
                            translation_data = json.loads(line)
                            source_text = translation_data["translation"][language]
                            target_text = translation_data["translation"]["eng"]
                            
                            # Skipping empty translation or very long translation
                            if len(source_text) < 2 or len(target_text) < 2:
                                continue
                            if len(source_text.split(" ")) > 1023 or len(target_text.split(" ")) > 1023:
                                continue

                            source_file.write(source_text + '\n')
                            target_file.write(target_text + '\n')
                            read += 1
                            if category == "train" and read == train_size:
                                break

def _convert_train_dataset_from_txt(source_language, target_language, train_dataset, train_size):
    for category in CATEGORIES:
        if category == "train":
            source_file_name = "train_{}_{}.txt".format(source_language, train_size)
            target_file_name = "train_{}_{}.txt".format(target_language, train_size)
            source_txt_name = "train_{}.txt".format(source_language)
            target_txt_name = "train_{}.txt".format(target_language)

            directory = os.path.join(DATASET_FOLDER_PATH, "{}/{}".format(source_language, category))

            for named_dataset in os.listdir(directory):
                dataset_path = os.path.join(directory, named_dataset)

                if os.path.basename(named_dataset) == train_dataset:
                    source_file_path = os.path.join(dataset_path, source_file_name)
                    target_file_path = os.path.join(dataset_path, target_file_name)

                    # Cannot find .txt file in the folder
                    txt_source_path = os.path.join(dataset_path, source_txt_name)
                    txt_target_path = os.path.join(dataset_path, target_txt_name)
                    
                    # Generate source and target train file
                    if not os.path.isfile(txt_source_path):
                        logging.error(f"Please put {source_txt_name} in directory {dataset_path}.")
                    if not os.path.isfile(txt_target_path):
                        logging.error(f"Please put {target_txt_name} in directory {dataset_path}.")
                    if os.path.exists(source_file_path) and os.path.exists(target_file_path):
                        logging.debug(f"{source_file_name} and {target_file_name} already exist, skipping.")
                    else:
                        logging.debug(f"Generating {source_file_name} and {target_file_name}...")
                        with open(txt_source_path, 'r') as txt_source_file:
                            data_source = txt_source_file.readlines()
                        with open(txt_target_path, 'r') as txt_target_file:
                            data_target = txt_target_file.readlines()

                        with open(source_file_path, 'w', encoding='utf-8') as source_file, open(target_file_path, 'w', encoding='utf-8') as target_file:
                            read = 0
                            for source_text, target_text in zip(data_source, data_target):
                                # Skipping empty translation or very long translation
                                if len(source_text) < 2 or len(target_text) < 2:
                                    continue
                                if len(source_text.split(" ")) > 1023 or len(target_text.split(" ")) > 1023:
                                    continue

                                source_file.write(source_text)
                                target_file.write(target_text)
                                read += 1
                                if read == train_size:
                                    break
        else:
            source_file_name = "{}_{}.txt".format(category, source_language)
            target_file_name = "{}_{}.txt".format(category, target_language)
            backup_source_file_name = "backup_{}".format(source_file_name)
            backup_target_file_name = "backup_{}".format(target_file_name)

            directory = os.path.join(DATASET_FOLDER_PATH, "{}/{}".format(source_language, category))

            for named_dataset in os.listdir(directory):
                dataset_path = os.path.join(directory, named_dataset)
                source_file_path = os.path.join(dataset_path, source_file_name)
                target_file_path = os.path.join(dataset_path, target_file_name)
                backup_source_file_path = os.path.join(dataset_path, backup_source_file_name)
                backup_target_file_path = os.path.join(dataset_path, backup_target_file_name)
                    
                # Check if file has been validated or not based on the backup file
                if not os.path.isfile(source_file_path):
                    logging.error(f"Please put {source_file_name} in directory {dataset_path}.")
                if not os.path.isfile(target_file_path):
                    logging.error(f"Please put {target_file_name} in directory {dataset_path}.")
                if os.path.exists(backup_source_file_path) and os.path.exists(backup_target_file_path):
                    logging.debug(f"Source/target {category} has been verified, skipping.")
                else:
                    logging.debug(f"Source/target {category} needs to be verified.")
                    # Rename to backup, then write backup to source/target
                    os.rename(source_file_path, backup_source_file_path)
                    os.rename(target_file_path, backup_target_file_path)
                    with open(backup_source_file_path, 'r') as txt_backup_source:
                        data_source = txt_backup_source.readlines()
                    with open(backup_target_file_path, 'r') as txt_backup_target:
                        data_target = txt_backup_target.readlines()

                    with open(source_file_path, 'w', encoding='utf-8') as source_file, open(target_file_path, 'w', encoding='utf-8') as target_file:
                        for source_text, target_text in zip(data_source, data_target):
                            # Skipping empty translation or very long translation
                            if len(source_text) < 2 or len(target_text) < 2:
                                continue
                            if len(source_text.split(" ")) > 1023 or len(target_text.split(" ")) > 1023:
                                continue

                            source_file.write(source_text)
                            target_file.write(target_text)
                            
def transform_dataset(source_language, target_language, train_dataset, train_size, data_type):
    if data_type == "txt":
        _convert_train_dataset_from_txt(source_language, target_language, train_dataset, train_size)
    elif data_type == "json":
        # TODO: change parameter to source_language, target_language
        _convert_dataset_from_json(target_language, train_dataset, train_size)
    else:
        raise ValueError("Data type is not recognized!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source_language', type=str, required=True, help="Language translate to")
    parser.add_argument('-o', '--target_language', type=str, required=True, help="Language translate from")
    parser.add_argument('-d', '--train_dataset', type=str, required=True, help="Train dataset name")
    parser.add_argument('-s', '--train_size', type=int, required=True, help="Train size")
    parser.add_argument('-t', '--original_data_type', type=str, required=True, help="Currently supports json and txt")
    args = parser.parse_args()

    transform_dataset(source_language=args.source_language, target_language=args.target_language,
                      train_dataset=args.train_dataset,
                      train_size=args.train_size, data_type=args.original_data_type)
