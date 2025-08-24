import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def preprocess_data(file_path):
    """Loads, tokenizes, and prepares data for LSTM model."""
    # Load the text data
    with open(file_path, 'r', encoding='utf-8') as f:
        faqs = f.read()

    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([faqs])
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences
    input_sequences = []
    for sentence in faqs.split('\n'):
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i+1])

    # Pad sequences
    max_len = max([len(x) for x in input_sequences])
    padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

    # Create predictors (X) and label (y)
    X = padded_input_sequences[:, :-1]
    y = padded_input_sequences[:, -1]

    # One-hot encode the label
    y = to_categorical(y, num_classes=total_words)

    return X, y, tokenizer, max_len, total_words