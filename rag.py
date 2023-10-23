"""
Retrieval-Augmented Generation (RAG) model for open-domain question answering.
Author (Research and Developed):  Teetouch Jaknamon, Natapong Nitarach
LICENSE: CC-BY-SA-3.0
"""

import unicodedata
import gc
import os
import re
import logging
import ctypes
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import faiss
from faiss import read_index
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
import torch

def clean_memory():
    """Clean up system and GPU memory."""
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

def retrival_template(example):
    options = ["context","category"]
    prompt_context = ["Represent this sentence for searching relevant passages:", example["instruction"]]

    prompt_context.extend([f"{opt}){example[opt]}" for opt in options])
    example['full_text'] = '\n'.join(prompt_context)
    return example

def construct_corpus(row):
    """Construct the corpus for the TF-IDF vectorizer."""
    if 'What is' in row.prompt:
        return ' '.join([row.prompt[7:-1] + ' is that ' + opt for opt in [row.A, row.B, row.C, row.D, row.E]])
    elif "Which of the following statements" in row.prompt:
        return (row.prompt.replace("Which of the following statements ", "") + " ".join([row.A, row.B, row.C, row.D, row.E])).replace("?", ". ")
    elif "Which of the following is" in row.prompt:
        return (row.prompt.replace("Which of the following is ", "") + " ".join([row.A, row.B, row.C, row.D, row.E])).replace("?", ". ")
    else:
        return str(row.prompt) + " " + " ".join([str(row.A), str(row.B), str(row.C), str(row.D), str(row.E)])

def retrieval_parallel(df_valid_chunk, modified_texts, stop_words):
    corpus_df_valid = df_valid_chunk.apply(construct_corpus, axis=1).values

    vectorizer1 = TfidfVectorizer(ngram_range=(1, 2),
                                  token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
                                  stop_words=stop_words)
    vectorizer1.fit(corpus_df_valid)
    vocab_df_valid = vectorizer1.get_feature_names_out()
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                 token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
                                 stop_words=stop_words,
                                 vocabulary=vocab_df_valid)
    
    print(len(modified_texts))
    vectorizer.fit(modified_texts[:500000])
    corpus_tf_idf = vectorizer.transform(corpus_df_valid)

    print(f"length of vectorizer vocab is {len(vectorizer.get_feature_names_out())}")

    chunk_size = 100000
    top_per_chunk = 10
    top_per_query = 10

    all_chunk_top_indices = []
    all_chunk_top_values = []

    for idx in tqdm(range(0, len(modified_texts), chunk_size)):
        wiki_vectors = vectorizer.transform(modified_texts[idx: idx + chunk_size])
        temp_scores = (corpus_tf_idf * wiki_vectors.T).toarray()
        chunk_top_indices = temp_scores.argpartition(-top_per_chunk, axis=1)[:, -top_per_chunk:]
        chunk_top_values = temp_scores[np.arange(temp_scores.shape[0])[:, np.newaxis], chunk_top_indices]

        all_chunk_top_indices.append(chunk_top_indices + idx)
        all_chunk_top_values.append(chunk_top_values)

    top_indices_array = np.concatenate(all_chunk_top_indices, axis=1)
    top_values_array = np.concatenate(all_chunk_top_values, axis=1)

    merged_top_scores = np.sort(top_values_array, axis=1)[:, -top_per_query:]
    merged_top_indices = top_values_array.argsort(axis=1)[:, -top_per_query:]
    articles_indices = top_indices_array[np.arange(top_indices_array.shape[0])[:, np.newaxis], merged_top_indices]

    return articles_indices, merged_top_scores


def SplitList(mylist, chunk_size):
    """Split a list into chunks."""
    return [mylist[offs:offs + chunk_size] for offs in range(0, len(mylist), chunk_size)]

def get_relevant_documents(df_valid, cohere_dataset_filtered, stop_words, model, FAISS_FOLDER, DATA_FOLDER, K_TOP):
    """Get the relevant documents for the validation set."""
    df_chunk_size = 800

    modified_texts = cohere_dataset_filtered.map(
        lambda example: {'temp_text': unicodedata.normalize("NFKD", f"{example['text']}").replace('"', "")},
        num_proc=2)["temp_text"]

    all_articles_indices = []
    all_articles_values = []
    chunks = [df_valid.iloc[idx: idx + df_chunk_size] for idx in range(0, df_valid.shape[0], df_chunk_size)]
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(retrieval_parallel, chunks, [modified_texts] * len(chunks)))

    for articles_indices, merged_top_scores in results:
        all_articles_indices.append(articles_indices)
        all_articles_values.append(merged_top_scores)

    article_indices_array = np.concatenate(all_articles_indices, axis=0)
    articles_values_array = np.concatenate(all_articles_values, axis=0).reshape(-1)

    top_per_query = article_indices_array.shape[1]
    articles_flatten = [
        (articles_values_array[index], unicodedata.normalize("NFKD", cohere_dataset_filtered[idx.item()]["text"]))
        for index, idx in enumerate(article_indices_array.reshape(-1))]
    retrieved_articles = SplitList(articles_flatten, top_per_query)
    return retrieved_articles

def get_top_context(example, retrieved_articles_parsed):
    """Get the top context for the validation set."""
    index = example['id']
    context = ""
    for text in retrieved_articles_parsed[index][::-1]:
        context += text[1]
        context += "\n"
    example['context'] = context
    return example

def main():
    MODEL_PATH = "embedding/finetune"
    N_BATCHES = 5
    MAX_CONTEXT = 3200
    MAX_LENGTH = 4096
    CSV_PATH = "/data/train.csv"
    FAISS_FOLDER = '/data/faissbatchall'
    DATA_FOLDER = '/data/wekiall6m'
    K_TOP = 15

    stop_words = ['each', 'you', 'the', 'use', 'used', 'where', 'themselves', 'nor', "it's", 'how', "don't", 'just', 'your',
                  'about', 'himself', 'with', "weren't", 'hers', "wouldn't", 'more', 'its', 'were', 'his', 'their', 'then',
                  'been', 'myself', 're', 'not', 'ours', 'will', 'needn', 'which', 'here', 'hadn', 'it', 'our', 'there',
                  'than', 'most', "couldn't", 'both', 'some', 'for', 'up', 'couldn', "that'll", "she's", 'over', 'this',
                  'now', 'until', 'these', 'few', 'haven', 'of', 'wouldn', 'into', 'too', 'to', 'very', 'shan', 'before',
                  'the', 'they', 'between', "doesn't", 'are', 'was', 'out', 'we', 'me', 'after', 'has', "isn't", 'have',
                  'such', 'should', 'yourselves', 'or', 'during', 'herself', 'doing', 'in', "shouldn't", "won't", 'when',
                  'do', 'through', 'she', 'having', 'him', "haven't", 'against', 'itself', 'that', 'did', 'theirs', 'can',
                  'those', 'own', 'so', 'and', 'who', "you've", 'yourself', 'her', 'he', 'only', 'what', 'ourselves',
                  'again', 'had', "you'd", 'is', 'other', 'why', 'while', 'from', 'them', 'if', 'above', 'does', 'whom',
                  'yours', 'but', 'being', "wasn't", 'be']

    # load embed model
    model = SentenceTransformer(MODEL_PATH)

    # load data
    test = pd.read_csv(CSV_PATH)

    test = test[['prompt', 'A', 'B', 'C', 'D', 'E', 'answer']]
    test['id'] = test.index

    test = Dataset.from_pandas(test)
    test = test.map(retrival_template)
    logging.info(test)

    query_vector = model.encode(test['full_text'], normalize_embeddings=True, convert_to_tensor=True, device="cuda")
    query_vector = query_vector.detach().cpu().numpy()

    res = os.listdir(FAISS_FOLDER)
    res = sorted(res)
    logging.info(res)

    k = K_TOP
    dict_all = []
    res_gpu = faiss.StandardGpuResources()
    libc = ctypes.CDLL(None)

    # Loop through each FAISS index
    for data in res:
        # Extract the batch number from the index filename
        num_batch = re.findall(r'\d+', data)
        dataset_name = DATA_FOLDER + "/SciBatch" + str(num_batch[0])

        # Load the dataset
        ds = load_from_disk(dataset_name)
        print('read index...')

        # Load the FAISS index and convert to GPU
        index = faiss.read_index(f"{FAISS_FOLDER}/{data}")
        index = faiss.index_cpu_to_gpu(res_gpu, 0, index)  # Convert to GPU

        # Optionally, set nprobe for faster search (trade-off with accuracy)
        index.nprobe = 10

        # Search for the k-nearest neighbors
        distances, indices = index.search(query_vector, k)

        # Build the dictionary
        for distance_row, index_row in zip(distances, indices):
            inner_list = [
                {ds[int(idx)]['text']: dist}
                for idx, dist in zip(index_row, distance_row)
            ]
            dict_all.append(inner_list)

        # Cleanup
        del index, ds
        torch.cuda.empty_cache()


    text_list = []
    para_length = 14  # adjust

    for i in range(len(dict_all)):
        sorted_data = sorted(dict_all[i], key=lambda x: list(x.values())[0])
        keys_list = [key for dictionary in sorted_data[:para_length] for key in dictionary.keys()]
        for text in keys_list:
            text_list.append(text)

    text_list2 = (list(set(text_list)))
    logging.info(f'Number of unique paragraphs: {len(text_list2)}')

    del query_vector
    _ = gc.collect()
    libc.malloc_trim(0)

    text_dataframe = pd.DataFrame(text_list2, columns=['text'])
    cohere_dataset_filtered = Dataset.from_pandas(text_dataframe)

    logging.info(f'cohere_dataset_filtered: {cohere_dataset_filtered}')

    df_valid = pd.read_csv(CSV_PATH)
    df_valid = df_valid[['prompt', 'A', 'B', 'C', 'D', 'E', 'answer']]
    df_valid['id'] = df_valid.index

    retrieved_articles_parsed = get_relevant_documents(df_valid, cohere_dataset_filtered, stop_words, model,
                                                       FAISS_FOLDER, DATA_FOLDER, K_TOP)
    test = test.map(lambda x: get_top_context(x, retrieved_articles_parsed))

    df = test.to_pandas()
    df = df[["id", "prompt", "context", "A", "B", "C", "D", "E", "answer"]]

    df.to_csv(f"/result/{CSV_PATH}", index=False)

    # Cleanup
    clean_memory()

if __name__ == "__main__":
    main()
