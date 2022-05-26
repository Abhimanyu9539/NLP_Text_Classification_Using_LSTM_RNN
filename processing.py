import re
import config
import numpy as np
import pandas as pd
from tqdm import tqdm
from Source.utils import save_file
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder


def token_index(tokens, vocabulary, missing='<unk>'):
    """
    :param tokens: List of word tokens
    :param vocabulary: All words in the embeddings
    :param missing: Token for words not present in the vocabulary
    :return: List of integers representing the word tokens
    """
    idx_token = []
    for text in tqdm(tokens):
        idx_text = []
        for token in text:
            if token in vocabulary:
                idx_text.append(vocabulary.index(token))
            else:
                idx_text.append(vocabulary.index(missing))
        idx_token.append(idx_text)
    return idx_token


def main():

    # Read the glove vectors
    print("Processing embedding vectors...")
    with open(config.glove_vector_path, "rt",encoding="utf8") as f:
        emb = f.readlines()
    print(f"Vocabulary size: {len(emb)}")
    vocabulary, embeddings = [], []
    # Separate the embeddings from the vocabulary words
    for item in emb:
        vocabulary.append(item.split()[0])
        embeddings.append(item.split()[1:])
    # Convert embeddings from string to numpy float array
    embeddings = np.array(embeddings, dtype=np.float32)
    # Add embeddings for padding and unknown items
    vocabulary = ["<pad>", "<unk>"] + vocabulary
    embeddings = np.vstack([np.ones(50, dtype=np.float32), np.mean(embeddings, axis=0),
                            embeddings])
    # Save both the embeddings and vocabulary
    save_file(config.embeddings_path, embeddings)
    save_file(config.vocabulary_path, vocabulary)

    # Read the data file
    print("Processing data file...")
    data = pd.read_csv(config.data_path)
    # Drop rows where the text column is empty
    data.dropna(subset=[config.text_col_name], inplace=True)
    # Replace duplicate labels
    data.replace({config.label_col: config.product_map}, inplace=True)
    # Encode the label column
    label_encoder = LabelEncoder()
    label_encoder.fit(data[config.label_col])
    labels = label_encoder.transform(data[config.label_col])
    # Save the encoded labels
    save_file(config.labels_path, labels)
    save_file(config.label_encoder_path, label_encoder)

    # Process the text column
    input_text = data[config.text_col_name]
    # Convert text to lower case
    print("Converting text to lower case...")
    input_text = [i.lower() for i in tqdm(input_text)]
    # Remove punctuations except apostrophe
    print("Removing punctuations in text...")
    input_text = [re.sub(r"[^\w\d'\s]+", " ", i) for i in tqdm(input_text)]
    # Remove digits
    print("Removing digits in text...")
    input_text = [re.sub("\d+", "", i) for i in tqdm(input_text)]
    # Remove more than one consecutive instance of 'x'
    print("Removing 'xxxx...' in text")
    input_text = [re.sub(r'[x]{2,}', "", i) for i in tqdm(input_text)]
    # Replace multiple spaces with single space
    print("Removing additional spaces in text...")
    input_text = [re.sub(' +', ' ', i) for i in tqdm(input_text)]
    # Tokenize the text
    print("Tokenizing the text...")
    tokens = [word_tokenize(t) for t in tqdm(input_text)]
    # Take the first 20 tokens in each complaint text
    print("Taking the first 20 tokens of each complaint...")
    tokens = [i[:20] if len(i) > 19 else ['<pad>'] * (20 - len(i)) + i for i in tqdm(tokens)]
    # Convert tokens to integer indices from vocabulary
    print("Converting tokens to integer indices...")
    tokens = token_index(tokens, vocabulary)
    # Save the tokens
    save_file(config.tokens_path, tokens)


if __name__ == "__main__":
    main()
