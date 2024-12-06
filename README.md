# [ProxyLM: Predicting Language Model Performance on Multilingual Tasks via Proxy Models](https://arxiv.org/abs/2406.09334)

![framework for LM performance prediction](./logo.png)

Performance prediction is a method to estimate the performance of Language Models (LMs) on various Natural Language Processing (NLP) tasks, mitigating computational costs associated with model capacity and data for fine-tuning. Our paper presents ProxyLM, a scalable task- and language-agnostic framework designed to predict the performance of LMs using proxy models. These proxy models act as surrogates, approximating the performance of the LM of interest. By leveraging these proxy models, ProxyLM significantly reduces computational overhead in task evaluations, achieving up to a 37.08x speedup over traditional methods, even with our smallest proxy models. Our results across multiple multilingual NLP tasks and various robustness tests demonstrate that ProxyLM not only adapts well to previously unseen languages in pre-trained LMs, but also generalizes effectively across different datasets, outperforming the state-of-the-art by at least 1.78x in terms of root-mean-square error (RMSE).

If you are interested for more information, check out our [full paper](https://arxiv.org/abs/2406.09334).

## Contents

+ [Environment](#environment)
+ [Setup Instruction](#setup-instruction)
+ [Dataset Manual Download Links](#dataset-manual-download-links)
+ [LMs Manual Download Links](#lms-manual-download-links)
+ [Example LM Finetuning Usages](#example-lm-finetuning-usages)
+ [Example Regressor Usages](#example-regressor-usages)
+ [Citation](#citation)

## Environment

Python 3.10 or higher. Details of dependencies are in `setup.py`.

## Setup Instruction

1. Run `pip install .`. This will install basic dependencies to reproduce ProxyLM's framework. Note that the experimental records are in `src/proxy_regressor/csv_datasets`.

The following steps are **OPTIONAL** for finetuning the language models:

2. **[OPTIONAL]** In order to finetune the language models to produce more experimental records, please install additional dependencies through `pip install '.[fairseq, llama-factory]'` depending on the need. We use [fairseq](https://github.com/ritsukkiii/fairseq) to finetune/do inference on Machine Translation (MT), while we use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to finetune/do inference for intent classification and slot filling. 
3. **[OPTIONAL]** Specifically for MT, run `bash setup_mt_finetune.sh` which will automatically download selected models and our curated dataset for MT. If the model or dataset cannot be downloaded successfully, please refer to section [Dataset Manual Download Links](#dataset-manual-download-links) and [LMs Manual Download Links](#lms-manual-download-links).
4. **[OPTIONAL]** Specifically for intent classification and slot filling, replace `dataset_info.json` in `data` at installed LLaMA-Factory library with our version at `src/llama_factory_configs/dataset_info.json` to use MASSIVE dataset.

## Dataset Manual Download Links

1. Download our curated [dataset](https://drive.google.com/file/d/1RS_OjFxr6XsSAJ_JB0x16QmR1W4uD8tc/view) for LM fine-tuning. You can also download the dataset from the original papers of [MT560 dataset](https://aclanthology.org/2021.acl-demo.37) and [NusaTranslation dataset](https://aclanthology.org/2023.ijcnlp-main.60/), but we have compiled our dataset in a way that it smoothly runs within our pipeline.
2. Unzip the dataset by running `tar -xzvf dataset.tar.gz dataset` and put the `dataset` folder in `experiments` folder (need to be created) in the same directory as this `README.md`.

## LMs Manual Download Links

If any of the download link has expired or become invalid, please use the following link below to download the model manually.

+ [SMaLL100](https://github.com/alirezamshi/small100)
+ [M2M100 (1.2B)](https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100)
+ [NLLB-200 Distilled (1.3B)](https://github.com/facebookresearch/fairseq/tree/nllb)

## Example MT Finetuning Usages

+ Start training/finetuning tasks by running:
    ```bash
    python -m src.mt_finetune.<lm_name>.main --src_lang ${src_lang} --tgt_lang ${tgt_lang} --finetune 1 --dataset ${dataset} --size ${size}
    ```

+ Start generation tasks by running:
    ```bash
    python -m src.mt_finetune.<lm_name>.main --src_lang ${src_lang} --tgt_lang ${tgt_lang} --finetune 0
    ```

+ Replace `<lm_name>` with LMs name such as `m2m100`, `nllb`, `small100`, or `transformer`.
+ All the results will be displayed in `experiments` folder

## Example Intent Classification or Slot Filling Finetuning/Inference Usages

+ Start training/finetuning tasks by running:
    ```bash
    llamafactory-cli train <lm_name>_finetune_<task_name>.yaml
    ```

+ Start generation tasks by running:
    ```bash
    llamafactory-cli train <lm_name>_predict_<task_name>.yaml
    ```

+ Replace `<lm_name>` with LMs name such as `aya`, `llama3`, `smollm_135m`, `smollm_360m`, and `bloomz_560m`.
+ Replace `<task_name>` with proper task such as `intent` and `slot`.
+ The configs can be found in `src/llama_factory_configs` folders.

## Example Regressor Usages

+ General script to Run Experiment
    ```bash
    proxylm-cli --config <config.yaml>
    ```

+ Replace `<config.yaml>` with a proper YAML file, example YAMLs can be found in `sample_yaml` folder.
+ Fill `exp_mode` with either `random` or `lolo`. Specifically for MT task, `unseen`, `cross_dataset`, and `incremental` are valid.
+ Fill `regressor` with either `xgb`, `lgbm`, `poly`, or `mf`.
+ Fill `regressor_config` with a proper path to the regressor JSON.
+ Fill `score` with proper score to be used.
+ Fill `dataset_name` with proper dataset corresponding to the tasks (either `mt560`, `nusa`, `intent`, or `slot`).
+ Fill `model` with proper model corresponding to the tasks. For MT, use either `m2m100` or `nllb`. For intent or slot, use either `aya` or `llama3`
+ Specifically for `lolo` in `exp_mode`, you need to supply language to be left-out using `lang` argument. If you'd like to run `lolo` for all languages, supply `lang` with `all`.

## Citation

<u>If you use this code for your research, please cite the following work:</u>

```bibtex
@article{anugraha2024proxylm,
  title={ProxyLM: Predicting Language Model Performance on Multilingual Tasks via Proxy Models},
  author={Anugraha, David and Winata, Genta Indra and Li, Chenyue and Irawan, Patrick Amadeus and Lee, En-Shiun Annie},
  journal={arXiv preprint arXiv:2406.09334},
  year={2024}
}
```

If you have any questions, you can open a [GitHub Issue](https://github.com/davidanugraha/proxylm/issues) or send us an [email](mailto:david.anugraha@gmail.com).