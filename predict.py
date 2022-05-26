import re
import torch
import config
import argparse
from Source.utils import load_file
from nltk.tokenize import word_tokenize
from Source.model import RNNNetwork, LSTMNetwork


def main(args_):
    # Process input text
    input_text = args_.test_complaint
    input_text = input_text.lower()
    input_text = re.sub(r"[^\w\d'\s]+", " ", input_text)
    input_text = re.sub("\d+", "", input_text)
    input_text = re.sub(r'[x]{2,}', "", input_text)
    input_text = re.sub(' +', ' ', input_text)
    tokens = word_tokenize(input_text)
    # Add padding if the length of tokens is less than 20
    tokens = ['<pad>']*(20-len(tokens))+tokens
    # Load label encoder
    label_encoder = load_file(config.label_encoder_path)
    num_classes = len(label_encoder.classes_)
    # Load the model
    if args_.model_type == "lstm":
        model = LSTMNetwork(config.input_size, config.hidden_size,
                            num_classes)
        model_path = config.lstm_model_path
    else:
        model = RNNNetwork(config.input_size, config.hidden_size,
                           num_classes)
        model_path = config.rnn_model_path
    model.load_state_dict(torch.load(model_path))
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Load vocabulary and embeddings
    vocabulary = load_file(config.vocabulary_path)
    embeddings = load_file(config.embeddings_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Tokenize the input text
    idx_token = []
    for token in tokens:
        if token in vocabulary:
            idx_token.append(vocabulary.index(token))
        else:
            idx_token.append(vocabulary.index('<unk>'))
    # Pick the word embeddings for the tokens
    token_emb = embeddings[idx_token,:]
    # Convert token embeddings as a torch tensor
    inp = torch.from_numpy(token_emb)
    # Move the tensor to GPU if available
    inp = inp.to(device)
    # Create a batch of 1 data point
    inp = torch.unsqueeze(inp, 0)
    # Forward pass
    out = torch.squeeze(model(inp))
    # Find predicted class
    prediction = label_encoder.classes_[torch.argmax(out)]
    print(f"Predicted  Class: {prediction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_complaint", type=str, help="Test complaint")
    parser.add_argument("--model_type", type=str, default="rnn",
                        help="Model type: lstm or rnn")
    args = parser.parse_args()
    main(args)
