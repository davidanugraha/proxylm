import logging
import os
import subprocess

logging.basicConfig(level=logging.DEBUG)

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORM_DATA_SCRIPT_PATH = os.path.join(UTILS_DIR, "transform_dataset.py")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

EXP_DIR = os.path.join(ROOT_DIR, "experiments")
DATASET_FOLDER_NAME = "dataset"
DATASET_FOLDER_PATH = os.path.join(EXP_DIR, DATASET_FOLDER_NAME)

SRC_FINETUNE_DIR = os.path.join(ROOT_DIR, "src/lm_finetune")
SPM_SCRIPT_PATH = os.path.join(SRC_FINETUNE_DIR, "fairseq/scripts/spm_train.py")
ENCODER_SCRIPT_PATH = os.path.join(SRC_FINETUNE_DIR, "fairseq/scripts/spm_encode.py")
METRICS_DIR = os.path.join(SRC_FINETUNE_DIR, "metrics")
RUN_METRICS_SCRIPT_PATH = os.path.join(METRICS_DIR, "run_all_metrics.py")

def execute_command(command):
    logging.debug(f"Running {command}")
    
    subprocess_err_output = True
    if not subprocess_err_output:
        subprocess.run(command, shell=True, check=True)
    else:
        try:
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred {command}:")
            print("Exit code:", e.returncode)
            print("Output:", e.output.decode())  # Decoding may be necessary
            print("Error message:", e.stderr.decode())
    
def create_directories(directory):
    if os.path.exists(directory):
        # Removing a directory must be done manually to avoid accidental deletion after shedding blood on training
        err_msg = f"{directory} already present"
        logging.error(err_msg)
        raise FileExistsError(err_msg)

    os.makedirs(directory)
    os.chdir(directory)

    os.makedirs("encoded")
    os.makedirs("preprocess")
    os.makedirs("models")
    os.makedirs("outputs")

    os.chdir("models")
    os.makedirs("sentencepiece")
    os.makedirs("checkpoints")

    logging.debug(f"{directory} has been created")