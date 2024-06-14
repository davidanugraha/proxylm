import argparse
from collections import Counter
import os
import time

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer

from .transform_dataset import transform_dataset

import torch
import sentencepiece as spm
import spacy
import unicodedata
from sentence_transformers import SentenceTransformer

SPM_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flores200_spm.model")
PRETRAINED_SENTENCE_TRANSFORMER = 'paraphrase-multilingual-mpnet-base-v2'
DEVICE = torch.device('cuda')
NPROCS = os.cpu_count()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXP_DIR = os.path.join(ROOT_DIR, "experiments")
DATASET_FOLDER_NAME = "dataset"
DATASET_FOLDER_PATH = os.path.join(EXP_DIR, DATASET_FOLDER_NAME)

def preprocess_text_list(texts, lang, clean_tokens=True):
    # Normalize text using Unicode normalization (NFD form) to remove diacritics and special characters
    normalized_text_list = []
    processed_texts = []
    for text in texts:
        normalized_text = text.lower()
        normalized_text = unicodedata.normalize('NFD', normalized_text)
        normalized_text = ''.join(c for c in normalized_text if unicodedata.category(c) != 'Mn' and c.isalnum() or c.isspace())
        normalized_text_list.append(normalized_text)

    if lang == 'eng':
        # Use SpaCy to tokenize, remove stopwords, and lemmatized
        nlp = spacy.load('en_core_web_sm')
        for doc in nlp.pipe(normalized_text_list, n_process=NPROCS, batch_size=1000):
            if clean_tokens:
                result_tokens = [token.lemma_ for token in doc if not token.is_stop]
            else:
                result_tokens = [token for token in doc]
            processed_texts.append(result_tokens)
    else:
        # Tokenization use spm
        sp = spm.SentencePieceProcessor()
        sp.load(SPM_MODEL_PATH)
        for norm_text in normalized_text_list:
            processed_texts.append(sp.EncodeAsPieces(' '.join(norm_text)))
    
    return processed_texts

def preprocess_individual_text(text, lang, clean_tokens=True):
    # Normalize text using Unicode normalization (NFD form) to remove diacritics and special characters
    normalized_text = text.lower()
    normalized_text = unicodedata.normalize('NFD', normalized_text)
    normalized_text = ''.join(c for c in normalized_text if unicodedata.category(c) != 'Mn' and c.isalnum() or c.isspace())

    if lang == 'eng':
        # Use SpaCy to tokenize, remove stopwords, and lemmatized
        nlp = spacy.load('en_core_web_sm')
        doc = nlp.pipe(normalized_text)
        if clean_tokens:
            result_tokens = [token.lemma_ for token in doc if not token.is_stop]
        else:
            result_tokens = [token for token in doc]
        return result_tokens
    else:
        # Tokenization use spm
        sp = spm.SentencePieceProcessor()
        sp.load(SPM_MODEL_PATH)
        return sp.EncodeAsPieces(' '.join(normalized_text))

# Calculate JSD pairwise
def calculate_jsd(list_of_counters):
    # Combine all keys from the all distributions and sort them
    if len(list_of_counters) < 2:
        raise(RuntimeError)
    all_keys = set(list_of_counters[0].keys())
    for counter in list_of_counters[1:]:
        all_keys = all_keys | set(counter.keys())
    all_keys = sorted(all_keys)

    # Initialize aligned distributions with sorted keys
    aligned_distribution_list = []
    for counter in list_of_counters:
        aligned_distribution_list.append(np.array([counter.get(key, 0) / sum(counter.values()) for key in all_keys]))

    # Calculate JSD for each pair
    jsd_values = []
    for i in range(len(aligned_distribution_list)):
        for j in range(i + 1, len(aligned_distribution_list)):
            jsd_values.append(jensenshannon(aligned_distribution_list[i], aligned_distribution_list[j]))

    return tuple(jsd_values)

def calculate_tfidf(document1, document2, lang):
    def customized_tokenizer(text):
        # Cleaning token will have some issues when all are stop_words (cosine distance will be 0)
        return preprocess_individual_text(text, lang, clean_tokens=False)
    
    # We batch train document which will be larger 1 by 1 (we can definitely sample but let's go for more accurate measurement while we can!)
    # Assume document1 is larger, so need index to keep track document1's sentence
    finished = False
    doc1_index = 0
    cosine_distance_dict = {"sum": 0, "count": 0}
    while not finished:
        # If partial_document1 is larger, we index document1, otherwise, 
        if len(document1[doc1_index:]) < len(document2):
            partial_document1 = document1[doc1_index:]
            partial_document2 = document2[:len(partial_document1)]
            finished = True
        else:
            partial_document1 = document1[doc1_index:(doc1_index + len(document2))]
            partial_document2 = document2[:]
            doc1_index += len(partial_document2)

        # Combine documents into a single list
        documents = [" ".join(partial_document1), " ".join(partial_document2)]
        if len(documents[0]) > 0 and len(documents[1]) > 0:
            # Create the TF-IDF vectorizer with custom tokenizer
            vectorizer = TfidfVectorizer(tokenizer=customized_tokenizer)

            # Fit the vectorizer to the combined documents
            tfidf_matrix = vectorizer.fit_transform(documents)
            tfidf_matrix1 = tfidf_matrix[:1]
            tfidf_matrix2 = tfidf_matrix[1:]
            cosine_distance_dict["sum"] += cosine(np.ravel(np.mean(tfidf_matrix1, axis=0)), np.ravel(np.mean(tfidf_matrix2, axis=0)))
            cosine_distance_dict["count"] += 1
    
    return cosine_distance_dict["sum"] / cosine_distance_dict["count"]

# Main function on featurizing the dataset based on given row
def featurize_dataset(row):
    # Open dataset both train, dev, and test
    lang = row['source_lang'] if row['source_lang'] != "eng" else row['target_lang']
    train_path = f"{DATASET_FOLDER_PATH}/{lang}/train/{row['train_dataset']}/train_{row['target_lang']}_{row['dataset_size']}.txt"
    dev_dataset = "flores" if row['train_dataset'] == "cc_aligned" or row['train_dataset'].startswith("MT560") else row['train_dataset']
    dev_path = f"{DATASET_FOLDER_PATH}/{lang}/dev/{dev_dataset}/dev_{row['target_lang']}.txt"
    test_path = f"{DATASET_FOLDER_PATH}/{lang}/test/{row['test_dataset']}/test_{row['target_lang']}.txt"
    
    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL_PATH)
    
    try:
        with open(train_path, 'r', encoding='utf-8'):
            pass
    except FileNotFoundError:
        transform_dataset(row['source_lang'], row['target_lang'], row['train_dataset'], row['dataset_size'], "txt")
    
    # Ready to feature dataset of train, dev, and test
    with open(train_path, 'r', encoding='utf-8') as train_txt, open(dev_path, 'r', encoding='utf-8') as dev_txt, open(test_path, 'r', encoding='utf-8') as test_txt:
        train_sentences = [line.strip() for line in train_txt]
        dev_sentences = [line.strip() for line in dev_txt]
        test_sentences = [line.strip() for line in test_txt]
        
        # 0. Dataset size
        train_size = len(train_sentences)
        dev_size = len(dev_sentences)
        test_size = len(test_sentences)
        
        # Tokenize and count each token
        train_tokens_list = preprocess_text_list(train_sentences, lang)
        dev_tokens_list = preprocess_text_list(dev_sentences, lang)
        test_tokens_list = preprocess_text_list(test_sentences, lang)
        train_token_counters = [Counter(tokens) for tokens in train_tokens_list]
        dev_token_counters = [Counter(tokens) for tokens in dev_tokens_list]
        test_token_counters = [Counter(tokens) for tokens in test_tokens_list]
        unioned_train_counter = sum(train_token_counters, Counter())
        unioned_dev_counter = sum(dev_token_counters, Counter())
        unioned_test_counter = sum(test_token_counters, Counter())
        
        # 1. Vocab size
        train_word_vocab_size = len(unioned_train_counter)
        dev_word_vocab_size = len(unioned_dev_counter)
        test_word_vocab_size = len(unioned_test_counter)

        # 2. Average Sentence Length
        train_uncleaned_tokens_list = preprocess_text_list(train_sentences, lang, clean_tokens=False)
        dev_uncleaned_tokens_list = preprocess_text_list(dev_sentences, lang, clean_tokens=False)
        test_uncleaned_tokens_list = preprocess_text_list(test_sentences, lang, clean_tokens=False)
        avg_train_sentence_length = sum(len(tokens) for tokens in train_uncleaned_tokens_list) / train_size
        avg_dev_sentence_length = sum(len(tokens) for tokens in dev_uncleaned_tokens_list) / dev_size
        avg_test_sentence_length = sum(len(tokens) for tokens in test_uncleaned_tokens_list) / test_size

        # 3. Word Overlap
        union_train = set(unioned_train_counter.keys())
        union_dev = set(unioned_dev_counter.keys())
        union_test = set(unioned_test_counter.keys())
        train_dev_word_overlap = len(union_train.intersection(union_dev)) / (len(union_train) + len(union_dev))
        train_test_word_overlap = len(union_train.intersection(union_test)) / (len(union_train) + len(union_test))
        dev_test_word_overlap = len(union_dev.intersection(union_test)) / (len(union_dev) + len(union_test))

        # 4. Type-Token Ratio (TTR)
        ttr_train = train_word_vocab_size / sum(len(tokens) for tokens in train_tokens_list)
        ttr_dev = dev_word_vocab_size / sum(len(tokens) for tokens in dev_tokens_list)
        ttr_test = test_word_vocab_size / sum(len(tokens) for tokens in test_tokens_list)
        
        # 5. Type-Token Ratio Distance
        train_dev_ttr_distance = (1 - (ttr_train / ttr_dev))**2
        train_test_ttr_distance = (1 - (ttr_train / ttr_test))**2
        dev_test_ttr_distance = (1 - (ttr_dev / ttr_test))**2
        
        # 6. JSD
        train_dev_jsd, train_test_jsd, dev_test_jsd = calculate_jsd([unioned_train_counter, unioned_dev_counter, unioned_test_counter])
        
        # 7. TF-IDF
        train_dev_tf_idf = calculate_tfidf(train_sentences, dev_sentences, lang)
        train_test_tf_idf = calculate_tfidf(train_sentences, test_sentences, lang)
        if len(dev_sentences) > len(test_sentences):
            dev_test_tf_idf = calculate_tfidf(dev_sentences, test_sentences, lang)
        else:
            dev_test_tf_idf = calculate_tfidf(test_sentences, dev_sentences, lang)

        # 8. SentenceTransformer
        sentence_transformer_model = SentenceTransformer(PRETRAINED_SENTENCE_TRANSFORMER)
        train_embeddings = sentence_transformer_model.encode(train_sentences, device=DEVICE)
        dev_embeddings = sentence_transformer_model.encode(dev_sentences, device=DEVICE)
        test_embeddings = sentence_transformer_model.encode(test_sentences, device=DEVICE)
        train_dev_st_similarity = cosine(np.mean(train_embeddings, axis=0), np.mean(dev_embeddings, axis=0))
        train_test_st_similarity = cosine(np.mean(train_embeddings, axis=0), np.mean(test_embeddings, axis=0))
        dev_test_st_similarity = cosine(np.mean(dev_embeddings, axis=0), np.mean(test_embeddings, axis=0))
        
        return dev_size, test_size, \
               train_word_vocab_size, dev_word_vocab_size, test_word_vocab_size, \
               avg_train_sentence_length, avg_dev_sentence_length, avg_test_sentence_length, \
               train_dev_word_overlap, train_test_word_overlap, dev_test_word_overlap, \
               ttr_train, ttr_dev, ttr_test, \
               train_dev_ttr_distance, train_test_ttr_distance, dev_test_ttr_distance, \
               train_dev_jsd, train_test_jsd, dev_test_jsd, \
               train_dev_tf_idf, train_test_tf_idf, dev_test_tf_idf, \
               train_dev_st_similarity, train_test_st_similarity, dev_test_st_similarity

def add_dataset_attr(csv_input, csv_output):
    start_time = time.time()
    df = pd.read_csv(csv_input)
    new_feature_columns = ['dev_size', 'test_size',
                           'train_word_vocab_size', 'dev_word_vocab_size', 'test_word_vocab_size',
                           'avg_train_sentence_length', 'avg_dev_sentence_length', 'avg_test_sentence_length',
                           'train_dev_word_overlap', 'train_test_word_overlap', 'dev_test_word_overlap',
                           'ttr_train', 'ttr_dev', 'ttr_test',
                           'train_dev_ttr_distance', 'train_test_ttr_distance', 'dev_test_ttr_distance',
                           'train_dev_jsd', 'train_test_jsd', 'dev_test_jsd',
                           'train_dev_tf_idf', 'train_test_tf_idf', 'dev_test_tf_idf',
                           'train_dev_st_similarity', 'train_test_st_similarity', 'dev_test_st_similarity']
    df[new_feature_columns] = df.apply(featurize_dataset, axis=1, result_type='expand')
    df.to_csv(csv_output, index=False)
    end_time = time.time()
    print(end_time - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--csv_input', type=str, required=True, help="Path to CSV file to add dataset attributes")
    parser.add_argument('-o', '--csv_output', type=str, required=True, help="Path to CSV file to save")
    args = parser.parse_args()

    add_dataset_attr(args.csv_input, args.csv_output)